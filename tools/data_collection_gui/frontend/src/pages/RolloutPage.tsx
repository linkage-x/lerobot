import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../apiClient";
import { CheckpointBrowser, successRate } from "../shared/CheckpointBrowser";
import { assistedSuccessBlocked, terminalEventDriver } from "./rolloutAttribution";
import { carriedOverNotice } from "./rolloutCarryover";
import { LIVE_STATES, sessionAvailability, sessionNote } from "./rolloutSessionControls";
import { Metric, PageHeader, StatusDot } from "../shared/ui";
import type {
  Checkpoint,
  RolloutMode,
  RolloutLandmarks,
  RolloutOutcomeEntry,
  RolloutRtcMode,
  RolloutRtcSchedule,
  RolloutRun,
  RolloutRuntimeOptions,
  SceneResetRequest,
  TableAlignment,
  TableWindow,
  TaskLadder
} from "../types";
import { RolloutLandingMap } from "./RolloutLandingMap";
import { RolloutLiveViewer } from "./RolloutLiveViewer";
import { SceneResetPanel } from "./SceneResetPanel";
import { TableAlignmentPanel } from "./TableAlignmentPanel";

/**
 * Running a trained checkpoint on the real FR3.
 *
 * Its own page rather than a panel on Training because it is the only screen in this GUI whose
 * buttons move a robot, and because what it needs on screen at the moment it matters -- the
 * frames the policy is being fed, the safety-clamp count, one Stop -- has nothing in common
 * with what a training run needs.
 *
 * Three deliberate frictions, all of them about a failure that would otherwise be silent:
 *   1. A checkpoint whose contract disagrees with the rig cannot be started without an override.
 *   2. Any mode that moves the arm needs motion confirmed in the same interaction.
 *   3. A finished rollout asks how it went, once, while the operator still remembers.
 *
 * Move to start is the fourth control, and the one that is not a friction. The launcher homes
 * the arm once, before the runtime process exists, so every rollout after the first begins
 * wherever the previous one stopped. Ending the session to fix that costs a minute of policy
 * reload; this sends one word down the control channel the running process is already reading,
 * and it is enabled only between rollouts, which is the only window in which nothing else is
 * commanding the arm.
 */

const RTC_SCHEDULES: RolloutRtcSchedule[] = ["EXP", "LINEAR", "ONES", "ZEROS"];
const DEFAULT_TASK_PROMPT_PLACEHOLDER = "Pick up the peg and insert it fully into the hole.";

/** How many rollouts the history table opens on.
 *
 * It used to render the newest 40, which on a page whose real work sits above it meant the
 * card ran on for screens. The window keeps the scroll inside the card. */
const HISTORY_VISIBLE = 10;

function isRtcPolicy(policyType: string): boolean {
  const normalized = policyType.trim().toLowerCase().replace(/[\s._-]+/g, "");
  return (
    normalized === "pi0" ||
    normalized.startsWith("pi05") ||
    normalized.startsWith("pi0fast") ||
    normalized.startsWith("smolvla")
  );
}

function positiveNumberOr(value: string, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function optionalNumberOrNull(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function stateTone(state: RolloutRun["state"]): string {
  // "running" is the arm-is-moving-right-now dot, which homing is as much as rolling is;
  // "armed" is the it-could-move-at-any-moment one. The state text beside it says which.
  if (state === "rolling" || state === "homing" || state === "resetting") return "running";
  // `finishing` is "armed" and not "running": the arm has stopped, but the loop moves it to the
  // init pose on its way back to the gate, so this is not a state to reach into the cell in.
  if (state === "waiting" || state === "starting" || state === "finishing") return "armed";
  if (state === "error") return "error";
  if (state === "complete") return "complete";
  return "idle";
}

// Written into the note when an operator overrides the assisted-success block, so the claim it
// rests on ("the policy finished it; I only took the arm afterwards") is in the log next to the
// grade instead of only in their memory.
const ASSISTED_SUCCESS_MARK = "[操作者确认：终点由策略完成，接管仅用于收尾]";

export function RolloutPage() {
  const [run, setRun] = useState<RolloutRun | null>(null);
  const [modes, setModes] = useState<RolloutMode[]>([]);
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [selected, setSelected] = useState<Checkpoint | null>(null);
  // The checkpoint the last rollout ran, held as an id until the picker's listing arrives and
  // can turn it into the object the contract gates are checked against.
  const [restoreCheckpointId, setRestoreCheckpointId] = useState("");
  // The one checkpoint whose per-checkpoint defaults must not fire when it lands. Selecting a
  // checkpoint normally resets the prompt and the RTC knobs, because settings tuned for one
  // policy are the wrong ones for another -- but the restored checkpoint arrives carrying the
  // settings that were recorded against it, and resetting those would undo the carry-over one
  // render after it landed. Consumed once, so picking it again later behaves like any pick.
  const skipDefaultsForRef = useRef("");
  const [modeId, setModeId] = useState("smoke");
  const [confirmMotion, setConfirmMotion] = useState(false);
  const [overrideContract, setOverrideContract] = useState(false);
  const [moveToStart, setMoveToStart] = useState(true);
  const [maxSteps, setMaxSteps] = useState("300");
  const [taskPrompt, setTaskPrompt] = useState("");
  const [rtcMode, setRtcMode] = useState<RolloutRtcMode>("auto");
  const [rtcExecutionHorizon, setRtcExecutionHorizon] = useState("16");
  const [rtcMaxGuidanceWeight, setRtcMaxGuidanceWeight] = useState("10");
  const [rtcPrefixAttentionSchedule, setRtcPrefixAttentionSchedule] =
    useState<RolloutRtcSchedule>("EXP");
  const [rtcReplanQueueSize, setRtcReplanQueueSize] = useState("25");
  const [rtcInferenceDelaySteps, setRtcInferenceDelaySteps] = useState("");
  const [commandEmaAlpha, setCommandEmaAlpha] = useState("");
  // Off until a previous rollout says otherwise. Takeover opens a second action source onto a
  // loop that is moving a real arm, so when it does come back on the carry-over notice says so
  // out loud -- the switch itself lives in a subcard that is easy to start a rollout without
  // ever scrolling to.
  const [daggerTakeover, setDaggerTakeover] = useState(false);
  const [daggerRecord, setDaggerRecord] = useState(true);
  const [daggerDatasetRoot, setDaggerDatasetRoot] = useState("");
  const [daggerReleaseAfterS, setDaggerReleaseAfterS] = useState("");
  // Ticked only for the one case the trace cannot tell apart: a rollout the policy finished
  // before the operator took the arm to tidy up afterwards. Recorded in the note when used, so
  // the claim is in the log rather than only in the operator's memory.
  const [assistedSuccessAck, setAssistedSuccessAck] = useState(false);
  const [showRolloutAdvanced, setShowRolloutAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [outcomeNote, setOutcomeNote] = useState("");
  const [ladders, setLadders] = useState<TaskLadder[]>([]);
  const [outcomeTask, setOutcomeTask] = useState("");
  const [outcomeStageId, setOutcomeStageId] = useState("");
  // A list, because a rollout with three takeover spans has three reasons the operator reached
  // in. The stage stays one number -- it is an ordinal on a chain and two checkpoints have to be
  // comparable on it -- but "why did you have to help" is genuinely plural, and the reasons that
  // did not fit in the single field used to end up in the prose of the note where nothing counts
  // them. Order is the order they happened, so the first entry is the one that belongs to the
  // graded stage.
  const [outcomeBlockers, setOutcomeBlockers] = useState<string[]>([]);
  const [history, setHistory] = useState<RolloutOutcomeEntry[]>([]);
  const [showAllHistory, setShowAllHistory] = useState(false);
  const [landmarks, setLandmarks] = useState<RolloutLandmarks>({});
  const [frameNonce, setFrameNonce] = useState(0);
  const [backgroundNonce, setBackgroundNonce] = useState(0);
  const [tableAlignment, setTableAlignment] = useState<TableAlignment | null>(null);
  // The scene reset panel's own Reset button, handed up so the session bar can offer it without
  // the operator scrolling to the panel. The function is held in a ref rather than in state
  // because the panel republishes it on every render -- storing it in state would re-render this
  // page just as often -- and only the boolean the bar draws itself from is state.
  const sceneResetRunnableRef = useRef<(() => Promise<void>) | null>(null);
  const [sceneResetRunnable, setSceneResetRunnable] = useState(false);
  const onSceneResetRunnableChange = useCallback((runnable: (() => Promise<void>) | null) => {
    sceneResetRunnableRef.current = runnable;
    setSceneResetRunnable(runnable !== null);
  }, []);
  const logRef = useRef<HTMLPreElement | null>(null);

  const mode = useMemo(() => modes.find((item) => item.id === modeId), [modes, modeId]);
  // Read off the mode rather than assumed from `interactive`: the DAgger rehearsal is interactive
  // and takes the arm over, but it is its own program and takes none of these settings.
  const takeoverSupported = Boolean(mode?.takeover);
  const isLive = run !== null && LIVE_STATES.has(run.state);
  // Scene reset drives the arm through the rollout process's own stdin, so it is only offered
  // while that process is sitting between rollouts waiting for a command.
  const sceneResetPanelUsable = Boolean(run?.interactive) && run?.state === "waiting";
  // Every button in the session bar, decided in one place. The bar is pressable from a scroll
  // position where none of the run's state is on screen, so the rules cannot be spelled out
  // beside the thing each one reads.
  const availability = sessionAvailability(run, busy, sceneResetRunnable);
  const sceneResetBarReason = availability.canResetScene
    ? "Sends the reset the panel below is set up for: a new sample inside the painted region."
    : sceneResetPanelUsable
      ? "Paint a target region and tick the motion box in Scene reset below first."
      : run?.state === "finishing"
        // Named rather than folded into "between the rollouts", because this is the state the
        // operator presses in: the rollout has ended, the arm has stopped, and the runtime is
        // still writing. A reset sent now is dropped at the gate, so it is not offered.
        ? "The last rollout is still being written out. Reset comes back when the runtime reaches its next command."
        : "Only between the rollouts of an interactive session.";
  // The map is painted in base x/y, so the reference layer is the camera that looks across the
  // table rather than the one riding the gripper.
  const sceneResetCameraKey = run?.cameraKeys?.includes("side")
    ? "side"
    : run?.cameraKeys?.[0] ?? "side";
  // Handed to both maps so each asks for its own base-frame rectangle. Absent until the camera
  // has been aligned to the table: no backdrop is the correct drawing of "we do not know where
  // this picture is", and it is what the panel below exists to change.
  const tableViewUrl = useCallback(
    (window: TableWindow, width: number, height: number) =>
      api.tableViewUrl(sceneResetCameraKey, window, width, height, backgroundNonce),
    [sceneResetCameraKey, backgroundNonce]
  );
  const tableBackdrop = tableAlignment?.calibrated ? tableViewUrl : undefined;
  // The plane the demonstrations released the peg on is the plane the pegs and the landing
  // points live on, so it is the one worth projecting.
  const tablePlaneZ = landmarks.placeXyz?.[2] ?? 0.035;
  const tableCentre: [number, number] = landmarks.hole ??
    (landmarks.placeXyz ? [landmarks.placeXyz[0], landmarks.placeXyz[1]] : [0.45, 0.0]);
  const blocking = useMemo(
    () => (selected?.issues ?? []).filter((issue) => issue.level === "block"),
    [selected]
  );
  const isRtcCheckpoint = useMemo(
    () => isRtcPolicy(selected?.policyType ?? ""),
    [selected?.policyType]
  );
  const rolloutRuntimeOptions = useMemo<RolloutRuntimeOptions>(
    () => ({
      taskPrompt: taskPrompt.trim() || undefined,
      rtcMode,
      rtcExecutionHorizon: positiveNumberOr(rtcExecutionHorizon, 16),
      rtcMaxGuidanceWeight: positiveNumberOr(rtcMaxGuidanceWeight, 10),
      rtcPrefixAttentionSchedule,
      rtcReplanQueueSize: positiveNumberOr(rtcReplanQueueSize, 25),
      rtcInferenceDelaySteps: optionalNumberOrNull(rtcInferenceDelaySteps),
      commandEmaAlpha: optionalNumberOrNull(commandEmaAlpha),
      // Sent only for the modes the launcher forwards it to. On any other mode the gateway
      // refuses the start rather than dropping the setting, so not sending it is what keeps a
      // leftover switch from blocking a smoke test.
      daggerTakeover: takeoverSupported ? daggerTakeover : false,
      daggerRecord,
      daggerDatasetRoot: daggerDatasetRoot.trim() || undefined,
      daggerReleaseAfterS: optionalNumberOrNull(daggerReleaseAfterS)
    }),
    [
      taskPrompt,
      rtcMode,
      rtcExecutionHorizon,
      rtcMaxGuidanceWeight,
      rtcPrefixAttentionSchedule,
      rtcReplanQueueSize,
      rtcInferenceDelaySteps,
      commandEmaAlpha,
      takeoverSupported,
      daggerTakeover,
      daggerRecord,
      daggerDatasetRoot,
      daggerReleaseAfterS
    ]
  );

  // Only this checkpoint's rollouts. Two checkpoints' landing points on one map look like one
  // policy with twice the scatter, which is the opposite of what the map is for.
  const mappedEntries = useMemo(
    () =>
      history.filter(
        (entry) => entry.geometry && (!run?.checkpointId || entry.checkpointId === run.checkpointId)
      ),
    [history, run?.checkpointId]
  );

  // The log is served newest first, and the rows that answer "did that last run go in" are the
  // ones at the top. The rest stays one click away rather than several screens down.
  const visibleHistory = useMemo(
    () => (showAllHistory ? history : history.slice(0, HISTORY_VISIBLE)),
    [history, showAllHistory]
  );

  const refreshHistory = useCallback(async () => {
    setHistory(await api.fetchRolloutOutcomes());
  }, []);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      const payload = await api.fetchRolloutStatus();
      if (cancelled || !payload) return;
      setRun(payload.rollout);
      setModes(payload.modes);
      setTrainingBusy(payload.trainingBusy);
    };
    void tick();
    const timer = window.setInterval(tick, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  // Fetched once per dataset rather than on every poll: the first call reduces the dataset's
  // parquet down to one point per episode, and the answer only changes when the checkpoint under
  // test was trained on something else.
  useEffect(() => {
    const datasetRoot = run?.datasetRoot ?? "";
    if (!datasetRoot) return;
    if (landmarks.datasetRoot === datasetRoot) return;
    let cancelled = false;
    void (async () => {
      const payload = await api.fetchRolloutLandmarks();
      if (!cancelled && payload) setLandmarks(payload);
    })();
    return () => {
      cancelled = true;
    };
  }, [run?.datasetRoot, landmarks.datasetRoot]);

  // Once, on mount: the ladders are files in the repo, so they change when someone edits one
  // and not while a rollout is running.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const payload = await api.fetchTaskLadders();
      if (!cancelled && payload.length) setLadders(payload);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Seed the form from the last rollout, once, on mount. Tuning a policy means rolling out the
  // same settings against checkpoint after checkpoint, and retyping eight RTC knobs each time is
  // how they end up subtly different between two runs that were meant to be comparable.
  //
  // The gates (`confirmMotion`, `overrideContract`) are absent from the payload by construction
  // and are not seeded here either: they are re-answered for every start.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const params = await api.fetchRolloutLastParams();
      if (cancelled || !params || Object.keys(params).length === 0) return;
      if (params.mode) setModeId(params.mode);
      // Handed to the picker rather than applied here: this page has the id, and only the
      // listing has the checkpoint.
      if (params.checkpointId) {
        skipDefaultsForRef.current = params.checkpointId;
        setRestoreCheckpointId(params.checkpointId);
      }
      if (params.maxSteps !== undefined) setMaxSteps(String(params.maxSteps));
      if (params.moveToStart !== undefined) setMoveToStart(params.moveToStart);
      const options = params.runtimeOptions;
      if (options) {
        if (options.taskPrompt !== undefined) setTaskPrompt(options.taskPrompt);
        if (options.rtcMode !== undefined) setRtcMode(options.rtcMode);
        if (options.rtcExecutionHorizon !== undefined)
          setRtcExecutionHorizon(String(options.rtcExecutionHorizon));
        if (options.rtcMaxGuidanceWeight !== undefined)
          setRtcMaxGuidanceWeight(String(options.rtcMaxGuidanceWeight));
        if (options.rtcPrefixAttentionSchedule !== undefined)
          setRtcPrefixAttentionSchedule(options.rtcPrefixAttentionSchedule);
        if (options.rtcReplanQueueSize !== undefined)
          setRtcReplanQueueSize(String(options.rtcReplanQueueSize));
        // null is a real recorded value here -- "let the runtime estimate it" -- and must land
        // as an empty field rather than the string "null".
        setRtcInferenceDelaySteps(
          options.rtcInferenceDelaySteps == null ? "" : String(options.rtcInferenceDelaySteps)
        );
        setCommandEmaAlpha(
          options.commandEmaAlpha == null ? "" : String(options.commandEmaAlpha)
        );
        // The switch comes back with the destination and the handback. It is not one of the
        // motion gates -- it opens the SpaceMouse, it does not start the arm -- and a session
        // spent collecting corrections re-ticks it before every single rollout otherwise. What
        // it does get is a sentence in the notice, since it is a second action source and the
        // subcard holding it can sit unscrolled-to.
        if (options.daggerTakeover !== undefined) setDaggerTakeover(options.daggerTakeover);
        if (options.daggerRecord !== undefined) setDaggerRecord(options.daggerRecord);
        if (options.daggerDatasetRoot !== undefined)
          setDaggerDatasetRoot(options.daggerDatasetRoot);
        setDaggerReleaseAfterS(
          options.daggerReleaseAfterS == null ? "" : String(options.daggerReleaseAfterS)
        );
      }
      setNotice(carriedOverNotice(params));
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // The camera poll only runs while something is producing frames. Polling a finished rollout
  // would just 503 in a loop, and the gateway refuses stale frames anyway.
  useEffect(() => {
    if (!isLive) return undefined;
    const timer = window.setInterval(() => setFrameNonce((value) => value + 1), 200);
    return () => window.clearInterval(timer);
  }, [isLive]);

  // The map backdrop is a still, not a stream: the runtime publishes one frame each time it
  // parks the arm and waits. Re-fetching slowly is enough to pick up the frame from the
  // rollout that just ended, and polling it at the live camera rate would buy nothing --
  // each fetch also costs the gateway one perspective warp.
  useEffect(() => {
    if (!sceneResetPanelUsable) return undefined;
    setBackgroundNonce((value) => value + 1);
    const timer = window.setInterval(() => setBackgroundNonce((value) => value + 1), 3000);
    return () => window.clearInterval(timer);
  }, [sceneResetPanelUsable]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [run?.lastLines?.length]);

  // Motion confirmation is per-start, not sticky: it is a statement about this run, and
  // carrying it over to the next one would defeat the point of asking.
  useEffect(() => {
    setConfirmMotion(false);
  }, [modeId, selected?.id]);

  useEffect(() => {
    setOverrideContract(false);
  }, [selected?.id]);

  useEffect(() => {
    if (selected?.id && selected.id === skipDefaultsForRef.current) {
      skipDefaultsForRef.current = "";
      return;
    }
    setTaskPrompt("");
    setRtcMode("auto");
    setRtcExecutionHorizon("10");
    setRtcMaxGuidanceWeight("10");
    setRtcPrefixAttentionSchedule("EXP");
    setRtcReplanQueueSize("30");
    setRtcInferenceDelaySteps("");
    setCommandEmaAlpha("");
    setShowRolloutAdvanced(false);
  }, [selected?.id]);

  const wrap = async (label: string, action: () => Promise<{ ok: boolean; error?: string }>) => {
    setBusy(true);
    setError("");
    setNotice("");
    const result = await action();
    setBusy(false);
    if (!result.ok) setError(result.error || `${label} failed.`);
    return result;
  };

  const onStart = async () => {
    if (!selected || !mode) return;
    const result = await wrap("Start rollout", () =>
      api.startRollout({
        mode: mode.id,
        checkpointId: selected.id,
        confirmMotion,
        overrideContract,
        moveToStart,
        maxSteps: mode.id === "real_once" ? Number(maxSteps) || 300 : 0,
        runtimeOptions: rolloutRuntimeOptions
      })
    );
    if (result.ok) {
      setRun((result as { rollout?: RolloutRun }).rollout ?? null);
      setNotice(`${mode.label} started.`);
    }
  };

  const onControl = async (command: "start" | "stop" | "home" | "quit" | "takeover") => {
    const result = await wrap(`Rollout ${command}`, () => api.controlRollout(command));
    if (result.ok) setRun((result as { rollout?: RolloutRun }).rollout ?? null);
  };

  const onStop = async () => {
    const result = await wrap("Stop rollout", () => api.stopRollout());
    if (result.ok) setNotice("Stop sent.");
  };

  const onSceneReset = async (request: SceneResetRequest) => {
    const result = await wrap("Scene reset", () => api.resetRolloutScene(request));
    if (result.ok) {
      setRun((result as { rollout?: RolloutRun }).rollout ?? null);
      setNotice("Scene reset sent.");
    }
    return { ok: result.ok, error: result.error };
  };

  // One ladder ships today; the picker only appears once there are two, so the common case is
  // not a one-item menu.
  const ladder = useMemo(
    () => ladders.find((item) => item.task === outcomeTask) ?? ladders[0] ?? null,
    [ladders, outcomeTask]
  );
  const gradedStage = useMemo(
    () => ladder?.stages.find((stage) => stage.id === outcomeStageId) ?? null,
    [ladder, outcomeStageId]
  );
  // Shown to the operator before they commit, because this is the number the log will carry:
  // the outcome is *derived* from the stage rather than chosen beside it.
  const derivedOutcome =
    gradedStage && ladder ? (gradedStage.ordinal >= ladder.terminal ? "success" : "failure") : "";
  // Success means the policy got the object to the end of the task. If the operator was driving
  // when the object got there, it did not -- so that grade is blocked rather than warned about:
  // this log is what two checkpoints are compared on, and a success it did not earn is the one
  // entry that cannot be corrected by looking at more of them.
  // Measured by the runtime off its own per-step trace and printed on the rollout-end line as
  // `expert_spans=41-58;120-133`. Shown rather than left in the log because the grading rule
  // below refers to it: the stage a rollout earned is the one it reached before the *first*
  // takeover, and an operator asked to remember where that was, twenty minutes into a batch,
  // will grade against the last one instead.
  const takeoverSpans = run?.lastRolloutIntervention?.spans ?? [];
  const terminalWasOperators = terminalEventDriver(run?.lastRolloutGeometry) === "expert";
  const successBlocked = assistedSuccessBlocked(run?.lastRolloutGeometry, assistedSuccessAck);

  const onRecordOutcome = async (outcome?: "success" | "failure" | "aborted") => {
    const recordedSuccess = (outcome ?? derivedOutcome) === "success";
    const note =
      recordedSuccess && terminalWasOperators && assistedSuccessAck
        ? [ASSISTED_SUCCESS_MARK, outcomeNote].filter(Boolean).join(" ")
        : outcomeNote;
    const result = await wrap("Record outcome", () =>
      api.recordRolloutOutcome({
        note,
        ...(ladder && gradedStage ? { taskLadder: ladder.task, stageId: gradedStage.id } : {}),
        // Only meaningful on a shortfall: the terminal stage did not stop anywhere.
        ...(ladder && gradedStage && outcomeBlockers.length && gradedStage.ordinal < ladder.terminal
          ? { blockers: outcomeBlockers }
          : {}),
        // Sent only when it is not derivable. `aborted` is the one outcome a stage cannot
        // imply -- it says the round is not evidence about the policy at all.
        ...(outcome ? { outcome } : {})
      })
    );
    if (result.ok) {
      setOutcomeNote("");
      setOutcomeStageId("");
      setOutcomeBlockers([]);
      setAssistedSuccessAck(false);
      setNotice(`Recorded ${outcome ?? derivedOutcome}${gradedStage ? ` at stage ${gradedStage.ordinal}` : ""}.`);
      await refreshHistory();
    }
  };

  const startDisabled =
    busy ||
    isLive ||
    trainingBusy ||
    !selected ||
    !mode ||
    (mode.movesArm && !confirmMotion) ||
    (blocking.length > 0 && !overrideContract);

  return (
    // `rollout-page` lays the cards out with margins instead of a grid track, which is what lets
    // the bar below stick: a sticky grid item is measured against its own grid area, and an
    // auto-sized row is exactly the item's height, so inside .page's grid it could not move.
    <div className="page rollout-page">
      <PageHeader
        title="Real-robot rollout"
        subtitle="Run a trained checkpoint on the FR3, and record how it went."
      />

      {/* ------------------------------------------------------ session bar --- */}
      {/* The controls a live session is driven by, on screen wherever the page is scrolled to.

          They used to sit in the head of the live card, which is the order an author writes in
          rather than the order an operator works in. Between two rollouts the loop is: grade the
          one that just ended, put the peg back, start the next -- and those were three different
          scroll positions, walked a few dozen times an afternoon, with Start at the far end of
          it every time. Scene reset was the worst of them: a full screen of map and form, two
          cards down, for what is one button once the region has been painted.

          So the buttons live here, at a place on the window instead of a place on the page, and
          Reset scene is the panel's own button rather than a second implementation of it. What
          stayed on the page is what is read rather than pressed: the instruments in the live
          card, the landing map, the history. */}
      {isLive && run && (
        <div className="rollout-session-bar">
          <div className="rollout-session-state">
            <StatusDot state={stateTone(run.state)} />
            <strong>{run.state}</strong>
            <span className="hint">
              rollout {run.rolloutIndex || "—"} · step{" "}
              {run.maxSteps ? `${run.step} / ${run.maxSteps}` : run.step}
            </span>
            {/* Only for the states an operator otherwise reads as a broken button: a policy load
                that takes a minute, and the two in which the arm is already carrying out a
                command of its own. */}
            {sessionNote(run) && <span className="hint">{sessionNote(run)}</span>}
            {/* A refusal has to arrive where the button was pressed. The banner carrying the
                whole of it is at the top of the page, which is exactly where the operator is
                not standing once these controls stopped living there. */}
            {error && (
              <span className="rollout-session-error" title={error}>
                {error}
              </span>
            )}
          </div>
          {/* Left to right in the order the loop uses them: home the arm, put the peg back,
              start. Start and Stop sit together because only ever one of the two is live --
              Start on `waiting`, Stop on `rolling` -- so they cannot be confused for each
              other. */}
          <div className="row-actions">
            {run.interactive && (
              <>
                <button
                  type="button"
                  onClick={() => void onControl("home")}
                  disabled={!availability.canHome}
                  title="A rollout that ran to its own end leaves the arm as displaced as one that was stopped."
                >
                  Move to start
                </button>
                <button
                  type="button"
                  onClick={() => {
                    // The boolean this button is enabled from is state; the function it fires is
                    // a ref. They are written together, but a click landing in the render between
                    // them would find the ref empty -- and a button that does nothing at all is
                    // the failure this whole path was just fixed for. Say so instead.
                    const runnable = sceneResetRunnableRef.current;
                    if (!runnable) {
                      setNotice("Scene reset is not ready yet — press it again in a moment.");
                      return;
                    }
                    void runnable();
                  }}
                  disabled={!availability.canResetScene}
                  title={sceneResetBarReason}
                >
                  Reset scene
                </button>
                <button
                  type="button"
                  onClick={() => void onControl("start")}
                  disabled={!availability.canStart}
                >
                  Start rollout
                </button>
                <button
                  type="button"
                  onClick={() => void onControl("stop")}
                  disabled={!availability.canStop}
                >
                  Stop rollout
                </button>
              </>
            )}
            <button
              type="button"
              className="danger rollout-session-end"
              onClick={() => void onStop()}
              disabled={!availability.canEnd}
            >
              End session
            </button>
          </div>
        </div>
      )}

      {error && <div className="banner banner-error">{error}</div>}
      {notice && !error && <div className="banner banner-ok">{notice}</div>}
      {trainingBusy && (
        <div className="banner banner-warn">
          A training run is using the GPU. Stop it on the Training page before rolling out — a
          policy starved of inference time still sends commands, just later than the arm expects
          them.
        </div>
      )}

      {/* --------------------------------------------------------- live run --- */}
      {run && run.state !== "idle" && (
        <section className="card rollout-live">
          {/* The session's buttons are in the bar above, not here: this card is several
              screens tall and they are needed from all of it. What stays is what the card is
              for -- which checkpoint is on the arm right now. */}
          <div className="card-head">
            <h3>
              <StatusDot state={stateTone(run.state)} /> {run.mode || "rollout"} ·{" "}
              {run.checkpointId || "—"}
            </h3>
          </div>

          <p className="hint">{run.message}</p>

          {/* First thing in the card while a grade is owed, ahead of the instruments below:
              the rollout it is asking about is over, the viewer and the camera strip are
              showing a scene that has stopped moving, and this is the only thing being asked
              of the operator. It used to sit under all of them, which put the one form that
              cannot be filled in later at the bottom of the deepest card on the page. */}
          {run.pendingOutcomeFor > 0 && (
            <div className="subcard outcome-prompt">
              <h4>
                How did rollout {run.pendingOutcomeFor} go?
                {run.lastRolloutIntervention?.intervened && (
                  <span className="pill pill-warn outcome-assisted-pill">
                    人工接管 {run.lastRolloutIntervention.expertSteps ?? 0} 步
                    {takeoverSpans.length > 1 ? ` · ${takeoverSpans.length} 段` : ""}
                  </span>
                )}
              </h4>
              <p className="hint">
                Recorded against {run.checkpointId}. This is the only thing that lets two
                checkpoints be compared honestly later.
              </p>
              {run.lastRolloutIntervention?.intervened && (
                <p className="hint warn">
                  这一轮你接管过 {run.lastRolloutIntervention.expertSteps ?? 0} 步
                  {takeoverSpans.length
                    ? `，分 ${takeoverSpans.length} 段：${takeoverSpans
                        .map(([first, last]) => `${first}–${last}`)
                        .join("、")}`
                    : ""}
                  ：
                  <strong>
                    要填，但不能按“孔插进去了”来评 ——
                    按<u>第一次接管之前</u>策略自己走到哪一步来评
                    {takeoverSpans.length > 1 ? `（也就是第 ${takeoverSpans[0][0]} 步之前）` : ""}。
                  </strong>
                  {takeoverSpans.length > 1
                    ? "第一次接管之后策略是从你摆好的状态往下走的，后面那些阶段不是它自己挣来的。阶段只填第一段之前够到的那一阶段；"
                    : "阶段选它被接管前够到的那一阶段；"}
                  “卡在哪”把<strong>每一段</strong>你之所以要伸手的原因都勾上（按顺序，第一个对应上面那一阶段），
                  note 里写清楚从第几步起是人在开。 终点是你的手放上去之后到达的，记成成功就是把这一分记在了
                  checkpoint 头上 —— 下一版是不是真的变好，比的就是这条记录。
                </p>
              )}
              <label className="field">
                <span>Note (optional)</span>
                <input
                  value={outcomeNote}
                  onChange={(event) => setOutcomeNote(event.target.value)}
                  placeholder="grasped but released early"
                />
              </label>
              {ladder && (
                <>
                  {ladders.length > 1 && (
                    <label className="field">
                      <span>Task</span>
                      <select
                        value={ladder.task}
                        onChange={(event) => {
                          setOutcomeTask(event.target.value);
                          setOutcomeStageId("");
                          setOutcomeBlockers([]);
                        }}
                      >
                        {ladders.map((item) => (
                          <option key={item.task} value={item.task}>
                            {item.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                  <label className="field">
                    <span>
                      走到了哪一步
                      {takeoverSpans.length > 1 ? "（第一次接管之前）" : ""}
                    </span>
                    <select
                      value={outcomeStageId}
                      onChange={(event) => setOutcomeStageId(event.target.value)}
                    >
                      <option value="">— 选择阶段 —</option>
                      {ladder.stages.map((stage) => (
                        <option key={stage.id} value={stage.id}>
                          {stage.ordinal} · {stage.label} — {stage.instance || stage.criterion}
                        </option>
                      ))}
                    </select>
                  </label>
                  {gradedStage && gradedStage.ordinal < ladder.terminal && (
                    <div className="field">
                      <span>卡在哪{takeoverSpans.length > 1 ? "（每段接管一个，按发生顺序）" : ""}</span>
                      {/* Checkboxes rather than a multiple <select>: this is filled in at the
                          rig, often with one hand, and a ctrl-click list is the control that
                          silently discards the previous choice when someone clicks without the
                          modifier. Order follows the order they were ticked, so the first one is
                          the reason belonging to the graded stage. */}
                      <div className="blocker-choices">
                        {ladder.blockers
                          .filter((blocker) => blocker.id !== "unknown")
                          .map((blocker) => (
                            <label key={blocker.id} className="checkbox">
                              <input
                                type="checkbox"
                                checked={outcomeBlockers.includes(blocker.id)}
                                onChange={() =>
                                  setOutcomeBlockers((current) =>
                                    current.includes(blocker.id)
                                      ? current.filter((id) => id !== blocker.id)
                                      : [...current, blocker.id]
                                  )
                                }
                              />
                              <span>
                                {outcomeBlockers.indexOf(blocker.id) >= 0
                                  ? `${outcomeBlockers.indexOf(blocker.id) + 1}. `
                                  : ""}
                                {blocker.label}
                                {blocker.instance ? ` — ${blocker.instance}` : ""}
                              </span>
                            </label>
                          ))}
                      </div>
                      <p className="hint">
                        {outcomeBlockers.length === 0
                          ? "一个都不勾 = 未判明。"
                          : `第 1 个（${outcomeBlockers[0]}）记作主因，对应上面那一阶段；其余按顺序一起存。`}
                      </p>
                    </div>
                  )}
                  <p className="hint">
                    成功 = 到达第 {ladder.terminal} 阶段（{ladder.stages[ladder.stages.length - 1].instance}）。
                    outcome 由阶段推出，不单独选 —— 两者能各填各的，就能互相矛盾。
                  </p>
                </>
              )}
              {terminalWasOperators && (!ladder || derivedOutcome === "success") && (
                <div className="subcard">
                  <p className="hint warn">
                    终点那一下是<strong>你</strong>开的（release 落在你的接管区间里），所以这一轮不能记成
                    success —— 记了就是把你的手算成了 checkpoint 的本事。选策略自己走到的那一阶段；
                    如果你伸手跟策略无关（碰倒了、复位、救硬件），用 Aborted。
                  </p>
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={assistedSuccessAck}
                      onChange={(event) => setAssistedSuccessAck(event.target.checked)}
                    />
                    <span>
                      任务在我接管前就已经完成，我只是接管去收尾（勾选后可以记 success，这句话会写进
                      note）
                    </span>
                  </label>
                </div>
              )}
              <div className="row-actions">
                {ladder ? (
                  <button
                    type="button"
                    onClick={() => void onRecordOutcome()}
                    disabled={busy || !gradedStage || (derivedOutcome === "success" && successBlocked)}
                  >
                    {!gradedStage
                      ? "Record (pick a stage first)"
                      : derivedOutcome === "success" && successBlocked
                        ? "Record (终点由人完成，不能记 success)"
                        : `Record stage ${gradedStage.ordinal} (${derivedOutcome})`}
                  </button>
                ) : (
                  <>
                    <button
                      type="button"
                      onClick={() => void onRecordOutcome("success")}
                      disabled={busy || successBlocked}
                    >
                      Success
                    </button>
                    <button
                      type="button"
                      onClick={() => void onRecordOutcome("failure")}
                      disabled={busy}
                    >
                      Failure
                    </button>
                  </>
                )}
                <button type="button" onClick={() => void onRecordOutcome("aborted")} disabled={busy}>
                  Aborted (not the policy&apos;s fault)
                </button>
              </div>
            </div>
          )}

          {/* There is no Take over button, and that is the design: moving the SpaceMouse takes
              the arm, and the policy resumes about a second after the operator stops. A button
              is a thing to find at the moment something is going wrong, and a latched one is a
              thing to forget -- the next rollout would start under a device nobody is holding.
              Who is driving right now is drawn in the live view's own pill, off the frames the
              runtime publishes, rather than off this click.

              Hold is the other half of that latch and does have a button, because a rollout this
              gateway launched holds the runtime's stdin as a pipe: the `t` key a terminal
              operator would press cannot reach it, so without this the browser has no brake. It
              freezes the arm at the last command it sent -- the gripper included -- which is what
              an operator wants when the policy is heading somewhere wrong and they need a moment
              before steering. The runtime clears the latch when the rollout stops, so it cannot
              be left on for the next one. */}
          {run.takeoverAvailable && (
            <div className="subcard">
              <div className="row-actions">
                <span className="pill">SpaceMouse armed</span>
                {run.state === "rolling" && (
                  <button
                    type="button"
                    onClick={() => void onControl("takeover")}
                    disabled={busy}
                  >
                    Hold / release (freeze the arm)
                  </button>
                )}
              </div>
              <p className="hint">
                Move the device to take the arm over.{" "}
                {run.daggerReleaseAfterS === 0
                  ? "Automatic handback is off: Hold is the only way in and out."
                  : run.daggerReleaseAfterS
                    ? `The policy resumes ${run.daggerReleaseAfterS} s after you stop.`
                    : "The policy resumes on its own once you stop."}{" "}
                Taking over does not move the gripper until you press a gripper button.
              </p>
              {run.daggerReportTimestamps && (
                <p className="hint">
                  Pre-flight: <code>report_timestamps={run.daggerReportTimestamps}</code> — the
                  driver dates each report, so a device that has gone quiet is told apart from one
                  still being pushed. Without it the arm would keep flying after your hand came
                  off; a real rollout cannot start without it.
                </p>
              )}
              {run.daggerDatasetPath ? (
                <p className="hint">
                  Corrections: <code>{run.daggerDatasetPath}</code>
                  {run.daggerEpisodes ? ` — ${run.daggerEpisodes} episode(s) so far` : ""}. Stop
                  the rollout before QC or training so the dataset writer can finalize the last
                  parquet file.
                </p>
              ) : (
                <p className="hint warn">
                  Steer only — corrections are not being written anywhere.
                </p>
              )}
              {Boolean(run.daggerDroppedFrames) && (
                <p className="hint warn">
                  {run.daggerDroppedFrames} correction frame(s) dropped past the buffer cap: a
                  takeover ran longer than the runtime holds in memory, so the end of it is
                  missing from the dataset.
                </p>
              )}
            </div>
          )}

          {run.interactive && run.state === "waiting" && !run.armAtStart && (
            <p className="hint">
              The arm is where the last rollout left it. The dataset frame is anchored to the
              pose the episodes started from, so the next rollout would begin somewhere the
              policy was never shown — press <b>Move to start</b> first. The gripper is left
              exactly as it is: if it is still holding something, take it before homing.
            </p>
          )}

          {/* Drawn from the joint angles the runtime publishes each step, so it follows the arm
              rather than replaying it afterwards. Mounted only while something is producing
              frames: the canvas holds WebGL context and STL meshes, and an idle page has no
              reason to. */}
          {isLive && (
            // Mounted for the whole session, polling only while a rollout is actually
            // publishing: the canvas holds its WebGL context and meshes between rollouts (so the
            // arm does not vanish and reappear), and nothing is asked for while nothing moves.
            <RolloutLiveViewer live={run.state === "rolling"} rolloutIndex={run.rolloutIndex} />
          )}

          <div className="metric-row">
            <Metric label="State" value={run.state} />
            <Metric label="Step" value={run.maxSteps ? `${run.step} / ${run.maxSteps}` : run.step} />
            <Metric label="Rollout" value={run.rolloutIndex || "—"} />
            <Metric label="Command" value={run.commandStatus || "—"} />
            <Metric label="Step-limited" value={run.clampedSteps} />
            <Metric label="Leashed" value={run.leashedSteps} />
            <Metric label="Tool frame" value={run.targetFrameName || "—"} />
          </div>

          {run.clampedSteps > 0 && (
            <p className="hint">
              {run.clampedSteps} step(s) asked for more motion in a single tick than the step
              limit allows, measured against the policy's own previous command. A few is normal;
              a steady stream means the policy is asking for motion the demonstrations never
              contained.
            </p>
          )}

          {run.leashedSteps > 0 && (
            <p className="hint">
              {run.leashedSteps} step(s) hit the leash: the command ran further ahead of the
              measured pose than tracking lag explains. Unlike step-limiting, this points at the
              arm rather than the policy — something is blocking it, or it has stopped following.
            </p>
          )}

          {run.cameraKeys.length > 0 && (
            <div className="rollout-cameras">
              {run.cameraKeys.map((cameraKey) => (
                <figure key={cameraKey}>
                  <img
                    src={api.rolloutCameraUrl(cameraKey, frameNonce)}
                    alt={`policy input ${cameraKey}`}
                    onError={(event) => {
                      (event.target as HTMLImageElement).style.visibility = "hidden";
                    }}
                    onLoad={(event) => {
                      (event.target as HTMLImageElement).style.visibility = "visible";
                    }}
                  />
                  <figcaption>{cameraKey}</figcaption>
                </figure>
              ))}
              <p className="hint wide">
                These are the frames the policy is being fed — after cropping and resizing, not
                the raw camera. If one is black or stale, the policy is seeing that too.
              </p>
            </div>
          )}

          {run.lastLines.length > 0 && (
            <pre className="log-block" ref={logRef}>
              {run.lastLines.join("\n")}
            </pre>
          )}
          {run.logPath && <p className="hint">Full log: <code>{run.logPath}</code></p>}
          {/* Named on screen because this is where the batch's evidence lands, and a batch
              analysed out of the wrong directory is worse than one nobody analysed. */}
          {run.tracePath && <p className="hint">Traces: <code>{run.tracePath}</code></p>}
        </section>
      )}

      {/* Above the map rather than below it: this is the other half of the loop the bar
          drives -- the region a reset samples from, and the pick pose it grasps at -- while
          the map is read afterwards. Its Reset button is published to the bar, so the panel
          is somewhere to come when the region has to change, not somewhere to visit between
          every pair of rollouts. */}
      <SceneResetPanel
        title="Scene reset"
        landmarks={landmarks}
        tableViewUrl={tableBackdrop}
        backgroundLabel={`${sceneResetCameraKey} camera`}
        backgroundHint={
          tableAlignment?.calibrated
            ? ""
            : `no ${sceneResetCameraKey} backdrop until the camera is aligned to the table below;`
        }
        busy={busy}
        disabled={!sceneResetPanelUsable}
        disabledReason={
          !run?.interactive
            ? "Start Interactive rollouts first; that process owns the FR3 connection."
            : run.state === "finishing"
              ? "The last rollout is still being written out; the runtime is not reading commands yet."
              : run.state !== "waiting"
                ? "Scene reset is only available between rollouts."
                : ""
        }
        onReset={onSceneReset}
        onRunnableChange={onSceneResetRunnableChange}
      />

      {/* ---------------------------------------------------- landing map --- */}
      <section className="card">
        <div className="card-head">
          <h3>Where the gripper landed</h3>
        </div>
        <RolloutLandingMap
          tableViewUrl={tableBackdrop}
          backgroundLabel={`${sceneResetCameraKey} camera`}
          landmarks={landmarks}
          entries={mappedEntries}
          pendingIndex={run?.pendingOutcomeFor ?? 0}
          pendingGeometry={run?.lastRolloutGeometry}
          checkpointId={run?.checkpointId ?? selected?.id ?? ""}
        />
      </section>

      <TableAlignmentPanel
        cameraKey={sceneResetCameraKey}
        planeZDefault={tablePlaneZ}
        centre={tableCentre}
        disabled={!sceneResetPanelUsable}
        disabledReason={
          !run?.interactive
            ? "Start Interactive rollouts first; that process owns the FR3 connection and the cameras."
            : run.state !== "waiting"
              ? "Probing a point is only available between rollouts."
              : ""
        }
        onAlignmentChange={setTableAlignment}
      />

      {/* ------------------------------------------------------ checkpoint --- */}
      <section className="card">
        <div className="card-head">
          <h3>Checkpoint</h3>
        </div>
        <CheckpointBrowser
          mode="picker"
          selectedId={selected?.id ?? ""}
          onSelect={setSelected}
          restoreId={restoreCheckpointId}
          disabled={isLive}
        />
      </section>

      {/* ------------------------------------------------------------ mode --- */}
      <section className="card">
        <div className="card-head">
          <h3>Mode</h3>
        </div>

        <div className="mujoco-mode-picker">
          {modes.map((item) => (
            <button
              key={item.id}
              type="button"
              className={item.id === modeId ? "active" : ""}
              onClick={() => setModeId(item.id)}
              disabled={isLive}
            >
              {item.label}
              {item.movesArm ? " ⚠" : ""}
            </button>
          ))}
        </div>
        {mode && <p className="hint">{mode.description}</p>}

        {mode?.id === "real_once" && (
          <label className="field">
            <span>Step limit</span>
            <input
              value={maxSteps}
              onChange={(event) => setMaxSteps(event.target.value)}
              inputMode="numeric"
              disabled={isLive}
            />
          </label>
        )}

        <div className="subcard rollout-runtime-options">
          <h4>pi0.5+LoRA first rollout defaults</h4>
          <p className="hint">
            Recommended first rollout: keep RTC mode on <code>auto</code>, execution horizon{" "}
            <code>16</code>, max guidance <code>10</code>, prefix attention <code>EXP</code>,
            replan queue <code>25</code>, inference delay <code>auto</code>, and command EMA{" "}
            <code>off</code>.
          </p>
          <p className="hint">
            {selected
              ? isRtcCheckpoint
                ? "This checkpoint is a flow/VLA policy; RTC auto will be enabled for smoother chunked execution."
                : `This checkpoint is ${selected.policyType}; RTC auto stays disabled for ACT-style policies.`
              : "Pick a checkpoint; these defaults are safe to leave unchanged."}
          </p>
          <label className="field">
            <span>Task prompt override</span>
            <input
              value={taskPrompt}
              onChange={(event) => setTaskPrompt(event.target.value)}
              placeholder={`auto from dataset task; e.g. ${DEFAULT_TASK_PROMPT_PLACEHOLDER}`}
              disabled={isLive}
            />
          </label>
          <div className="row-actions">
            <label className="field inline">
              <span>RTC mode</span>
              <select
                value={rtcMode}
                onChange={(event) => setRtcMode(event.target.value as RolloutRtcMode)}
                disabled={isLive}
              >
                <option value="auto">auto (recommended)</option>
                <option value="enabled">force enabled</option>
                <option value="disabled">disabled</option>
              </select>
            </label>
            <label className="field inline">
              <span>Horizon</span>
              <input
                value={rtcExecutionHorizon}
                onChange={(event) => setRtcExecutionHorizon(event.target.value)}
                inputMode="numeric"
                disabled={isLive}
              />
            </label>
            <label className="field inline">
              <span>Guidance</span>
              <input
                value={rtcMaxGuidanceWeight}
                onChange={(event) => setRtcMaxGuidanceWeight(event.target.value)}
                inputMode="decimal"
                disabled={isLive}
              />
            </label>
          </div>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={showRolloutAdvanced}
              onChange={(event) => setShowRolloutAdvanced(event.target.checked)}
              disabled={isLive}
            />
            <span>Show advanced rollout knobs</span>
          </label>
          {showRolloutAdvanced && (
            <>
              <div className="row-actions">
                <label className="field inline">
                  <span>Attention schedule</span>
                  <select
                    value={rtcPrefixAttentionSchedule}
                    onChange={(event) =>
                      setRtcPrefixAttentionSchedule(event.target.value as RolloutRtcSchedule)
                    }
                    disabled={isLive}
                  >
                    {RTC_SCHEDULES.map((schedule) => (
                      <option key={schedule} value={schedule}>{schedule}</option>
                    ))}
                  </select>
                </label>
                <label className="field inline">
                  <span>Replan queue</span>
                  <input
                    value={rtcReplanQueueSize}
                    onChange={(event) => setRtcReplanQueueSize(event.target.value)}
                    inputMode="numeric"
                    disabled={isLive}
                  />
                </label>
                <label className="field inline">
                  <span>Delay steps</span>
                  <input
                    value={rtcInferenceDelaySteps}
                    onChange={(event) => setRtcInferenceDelaySteps(event.target.value)}
                    inputMode="numeric"
                    placeholder="auto"
                    disabled={isLive}
                  />
                </label>
                <label className="field inline">
                  <span>Command EMA</span>
                  <input
                    value={commandEmaAlpha}
                    onChange={(event) => setCommandEmaAlpha(event.target.value)}
                    inputMode="decimal"
                    placeholder="off"
                    disabled={isLive}
                  />
                </label>
              </div>
              <p className="hint">
                EMA is intentionally off for the first pi0.5+LoRA rollout: RTC replanning already
                smooths the queue, while extra EMA can blur the final insertion correction.
              </p>
            </>
          )}
        </div>

        {takeoverSupported && (
          <div className="subcard rollout-runtime-options">
            <h4>DAgger takeover</h4>
            <p className="hint">
              Moving the SpaceMouse takes the arm mid-rollout; the policy resumes on its own once
              you stop. There is no engage button — the device is the switch. Each stretch you
              drove becomes one episode flagged <code>is_intervention</code>, written after the
              rollout ends rather than at the moment you let go.
            </p>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={daggerTakeover}
                onChange={(event) => setDaggerTakeover(event.target.checked)}
                disabled={isLive}
              />
              <span>
                Open a SpaceMouse for this session (a second action source onto a moving arm —
                somebody has to be at the rig)
              </span>
            </label>
            {daggerTakeover && (
              <>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={daggerRecord}
                    onChange={(event) => setDaggerRecord(event.target.checked)}
                    disabled={isLive}
                  />
                  <span>
                    Keep the corrections as training data (uncheck only to feel out the handoff —
                    takeover still steers the arm, but nothing is written)
                  </span>
                </label>
                {daggerRecord ? (
                  <label className="field">
                    <span>Corrections dataset</span>
                    <input
                      value={daggerDatasetRoot}
                      onChange={(event) => setDaggerDatasetRoot(event.target.value)}
                      placeholder="auto: outputs/datasets/dagger_<checkpoint>, extended across sessions"
                      disabled={isLive}
                    />
                  </label>
                ) : (
                  <p className="hint warn">
                    Steer only: this session&apos;s corrections are discarded when it ends.
                  </p>
                )}
                <label className="field inline">
                  <span>Hand back after (s)</span>
                  <input
                    value={daggerReleaseAfterS}
                    onChange={(event) => setDaggerReleaseAfterS(event.target.value)}
                    inputMode="decimal"
                    placeholder="1"
                    disabled={isLive}
                  />
                </label>
                <p className="hint">
                  Seconds of a still device before the policy takes the arm back. <code>0</code>{" "}
                  turns automatic handback off, leaving <b>Hold</b> as the only way in and out.
                </p>
              </>
            )}
          </div>
        )}

        {mode?.movesArm && (
          <label className="checkbox">
            <input
              type="checkbox"
              checked={moveToStart}
              onChange={(event) => setMoveToStart(event.target.checked)}
              disabled={isLive}
            />
            <span>
              Home the arm first (the dataset frame is anchored to the pose episodes started from
              — skipping this places the whole trajectory somewhere else)
            </span>
          </label>
        )}

        {selected && blocking.length > 0 && (
          <div className="banner banner-error">
            <strong>This checkpoint does not match the rig.</strong>
            <ul>
              {blocking.map((issue) => (
                <li key={issue.field}>{issue.message}</li>
              ))}
            </ul>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={overrideContract}
                onChange={(event) => setOverrideContract(event.target.checked)}
                disabled={isLive}
              />
              <span>I have read these and want to run it anyway</span>
            </label>
          </div>
        )}

        {mode?.movesArm && (
          <label className="checkbox confirm-motion">
            <input
              type="checkbox"
              checked={confirmMotion}
              onChange={(event) => setConfirmMotion(event.target.checked)}
              disabled={isLive}
            />
            <span>
              The cell is clear and I am at the rig. <strong>{mode.label} moves the arm.</strong>
            </span>
          </label>
        )}

        {selected && (
          <p className="hint">
            Will run <code>{selected.id}</code> ({selected.policyType}) with tool frame{" "}
            <code>{selected.contract.targetFrameName || "rig default"}</code> against dataset{" "}
            <code>{selected.datasetRepoId || "—"}</code>. Track record so far:{" "}
            {successRate(selected)}.
          </p>
        )}

        <button type="button" onClick={() => void onStart()} disabled={startDisabled}>
          {isLive ? "Rollout in progress" : `Start ${mode?.label ?? "rollout"}`}
        </button>
      </section>

      {/* --------------------------------------------------------- history --- */}
      <section className="card">
        <div className="card-head">
          <h3>Rollout history</h3>
          <button type="button" onClick={() => void refreshHistory()}>
            Refresh
          </button>
        </div>
        {history.length === 0 ? (
          <p className="hint">No rollouts recorded yet.</p>
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Checkpoint</th>
                  <th>Mode</th>
                  <th>Outcome</th>
                  <th>Steps</th>
                  <th>Assisted</th>
                  <th>Note</th>
                </tr>
              </thead>
              <tbody>
                {visibleHistory.map((entry, index) => (
                  <tr key={`${entry.recordedAt}-${index}`}>
                    <td>{entry.recordedAt.replace("T", " ").replace("+00:00", "Z")}</td>
                    <td>{entry.checkpointId}</td>
                    <td>{entry.mode || "—"}</td>
                    <td>
                      <span
                        className={`pill pill-${
                          entry.outcome === "success"
                            ? "ok"
                            : entry.outcome === "failure"
                              ? "error"
                              : "warn"
                        }`}
                      >
                        {entry.outcome}
                      </span>
                    </td>
                    <td>{entry.steps || "—"}</td>
                    <td>
                      {entry.intervened ? (
                        <span className="pill pill-warn" title="A human drove part of this rollout">
                          人工 {entry.expertSteps ?? 0} 步
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td>{entry.note || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {history.length > HISTORY_VISIBLE && (
          <div className="row-actions">
            <button type="button" onClick={() => setShowAllHistory((value) => !value)}>
              {showAllHistory ? `Show latest ${HISTORY_VISIBLE}` : `Show all ${history.length}`}
            </button>
            <span className="hint">
              {showAllHistory
                ? `All ${history.length} recorded rollouts, newest first.`
                : `Latest ${HISTORY_VISIBLE} of ${history.length}, newest first.`}
            </span>
          </div>
        )}
      </section>
    </div>
  );
}
