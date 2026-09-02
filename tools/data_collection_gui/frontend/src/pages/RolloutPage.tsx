import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../apiClient";
import { CheckpointBrowser, successRate } from "../shared/CheckpointBrowser";
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

const LIVE_STATES = new Set(["starting", "waiting", "homing", "resetting", "rolling"]);
const RTC_SCHEDULES: RolloutRtcSchedule[] = ["EXP", "LINEAR", "ONES", "ZEROS"];
const DEFAULT_TASK_PROMPT_PLACEHOLDER = "Pick up the peg and insert it fully into the hole.";

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
  if (state === "waiting" || state === "starting") return "armed";
  if (state === "error") return "error";
  if (state === "complete") return "complete";
  return "idle";
}

export function RolloutPage() {
  const [run, setRun] = useState<RolloutRun | null>(null);
  const [modes, setModes] = useState<RolloutMode[]>([]);
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [selected, setSelected] = useState<Checkpoint | null>(null);
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
  const [showRolloutAdvanced, setShowRolloutAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [outcomeNote, setOutcomeNote] = useState("");
  const [ladders, setLadders] = useState<TaskLadder[]>([]);
  const [outcomeTask, setOutcomeTask] = useState("");
  const [outcomeStageId, setOutcomeStageId] = useState("");
  const [outcomeBlocker, setOutcomeBlocker] = useState("");
  const [history, setHistory] = useState<RolloutOutcomeEntry[]>([]);
  const [landmarks, setLandmarks] = useState<RolloutLandmarks>({});
  const [frameNonce, setFrameNonce] = useState(0);
  const [backgroundNonce, setBackgroundNonce] = useState(0);
  const [tableAlignment, setTableAlignment] = useState<TableAlignment | null>(null);
  const logRef = useRef<HTMLPreElement | null>(null);

  const mode = useMemo(() => modes.find((item) => item.id === modeId), [modes, modeId]);
  const isLive = run !== null && LIVE_STATES.has(run.state);
  // Scene reset drives the arm through the rollout process's own stdin, so it is only offered
  // while that process is sitting between rollouts waiting for a command.
  const sceneResetPanelUsable = Boolean(run?.interactive) && run?.state === "waiting";
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
      commandEmaAlpha: optionalNumberOrNull(commandEmaAlpha)
    }),
    [
      taskPrompt,
      rtcMode,
      rtcExecutionHorizon,
      rtcMaxGuidanceWeight,
      rtcPrefixAttentionSchedule,
      rtcReplanQueueSize,
      rtcInferenceDelaySteps,
      commandEmaAlpha
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
      }
      setNotice("Settings carried over from the last rollout. Motion confirmation is not.");
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

  const onRecordOutcome = async (outcome?: "success" | "failure" | "aborted") => {
    const result = await wrap("Record outcome", () =>
      api.recordRolloutOutcome({
        note: outcomeNote,
        ...(ladder && gradedStage ? { taskLadder: ladder.task, stageId: gradedStage.id } : {}),
        // Only meaningful on a shortfall: the terminal stage did not stop anywhere.
        ...(ladder && gradedStage && outcomeBlocker && gradedStage.ordinal < ladder.terminal
          ? { blocker: outcomeBlocker }
          : {}),
        // Sent only when it is not derivable. `aborted` is the one outcome a stage cannot
        // imply -- it says the round is not evidence about the policy at all.
        ...(outcome ? { outcome } : {})
      })
    );
    if (result.ok) {
      setOutcomeNote("");
      setOutcomeStageId("");
      setOutcomeBlocker("");
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
    <div className="page">
      <PageHeader
        title="Real-robot rollout"
        subtitle="Run a trained checkpoint on the FR3, and record how it went."
      />

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
          <div className="card-head">
            <h3>
              <StatusDot state={stateTone(run.state)} /> {run.mode || "rollout"} ·{" "}
              {run.checkpointId || "—"}
            </h3>
            {isLive && (
              <div className="row-actions">
                {run.interactive && (
                  <>
                    {/* Enabled only on `waiting`, which the runtime declares by printing
                        `interactive_waiting_for_start`. Everything before that -- homing the
                        arm, a minute of loading the policy, opening the cameras -- is
                        `starting`, and a Start pressed there is read by the listener thread and
                        then cleared by the loop when it reaches its wait. The click looked like
                        it worked and nothing happened. */}
                    <button
                      type="button"
                      onClick={() => void onControl("start")}
                      disabled={busy || run.state !== "waiting"}
                    >
                      Start rollout
                    </button>
                    <button
                      type="button"
                      onClick={() => void onControl("stop")}
                      disabled={busy || run.state !== "rolling"}
                    >
                      Stop rollout
                    </button>
                    {/* After Stop rather than beside Start, because that is the order it is
                        used in. Enabled on `waiting` rather than on "Stop was pressed": a
                        rollout that ran to its own end leaves the arm just as displaced as one
                        that was interrupted, and both land here. */}
                    <button
                      type="button"
                      onClick={() => void onControl("home")}
                      disabled={busy || run.state !== "waiting"}
                    >
                      Move to start
                    </button>
                  </>
                )}
                <button type="button" className="danger" onClick={() => void onStop()} disabled={busy}>
                  End session
                </button>
              </div>
            )}
          </div>

          <p className="hint">{run.message}</p>

          {/* There is no Take over button, and that is the design: moving the SpaceMouse takes
              the arm, and the policy resumes about a second after the operator stops. A button
              is a thing to find at the moment something is going wrong, and a latched one is a
              thing to forget -- the next rollout would start under a device nobody is holding.
              Who is driving right now is drawn in the live view's own pill, off the frames the
              runtime publishes, rather than off this click. */}
          {run.takeoverAvailable && run.state === "rolling" && (
            <p className="hint">
              SpaceMouse armed: move it to take the arm over. The policy resumes on its own once
              you stop.
            </p>
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

          {run.pendingOutcomeFor > 0 && (
            <div className="subcard outcome-prompt">
              <h4>How did rollout {run.pendingOutcomeFor} go?</h4>
              <p className="hint">
                Recorded against {run.checkpointId}. This is the only thing that lets two
                checkpoints be compared honestly later.
              </p>
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
                          setOutcomeBlocker("");
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
                    <span>走到了哪一步</span>
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
                    <label className="field">
                      <span>卡在哪</span>
                      <select
                        value={outcomeBlocker}
                        onChange={(event) => setOutcomeBlocker(event.target.value)}
                      >
                        <option value="">未判明</option>
                        {ladder.blockers
                          .filter((blocker) => blocker.id !== "unknown")
                          .map((blocker) => (
                            <option key={blocker.id} value={blocker.id}>
                              {blocker.label}
                              {blocker.instance ? ` — ${blocker.instance}` : ""}
                            </option>
                          ))}
                      </select>
                    </label>
                  )}
                  <p className="hint">
                    成功 = 到达第 {ladder.terminal} 阶段（{ladder.stages[ladder.stages.length - 1].instance}）。
                    outcome 由阶段推出，不单独选 —— 两者能各填各的，就能互相矛盾。
                  </p>
                </>
              )}
              <div className="row-actions">
                {ladder ? (
                  <button
                    type="button"
                    onClick={() => void onRecordOutcome()}
                    disabled={busy || !gradedStage}
                  >
                    {gradedStage
                      ? `Record stage ${gradedStage.ordinal} (${derivedOutcome})`
                      : "Record (pick a stage first)"}
                  </button>
                ) : (
                  <>
                    <button
                      type="button"
                      onClick={() => void onRecordOutcome("success")}
                      disabled={busy}
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
            : run.state !== "waiting"
              ? "Scene reset is only available between rollouts."
              : ""
        }
        onReset={onSceneReset}
      />

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
          <table className="table">
            <thead>
              <tr>
                <th>When</th>
                <th>Checkpoint</th>
                <th>Mode</th>
                <th>Outcome</th>
                <th>Steps</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody>
              {history.slice(0, 40).map((entry, index) => (
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
                  <td>{entry.note || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
