import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../apiClient";
import type {
  DependencyInstall,
  TrainingHistoryEntry,
  TrainingHost,
  TrainingMachine,
  TrainingRun,
  TrainingView,
  TrainingWandbStatus
} from "../types";
import { Metric, Modal, PageHeader, StatusDot } from "../shared/ui";
import { CheckpointBrowser } from "../shared/CheckpointBrowser";

// Mirrors KNOWN_POLICY_TYPES in tools/fr3/fr3_train_il_policy.py. Split by what has actually
// been trained and rolled out on the FR3 rig, because "selectable" and "someone has seen this
// work here" are different claims and the page should not blur them. Every type in both groups
// starts from its own upstream defaults -- the split says nothing about hyperparameters.
const VERIFIED_POLICIES = ["act", "diffusion"] as const;
const OTHER_POLICIES = [
  "vqbet",
  "tdmpc",
  "pi0",
  "pi0_fast",
  "pi05",
  "smolvla",
  "groot",
  "xvla",
  "wall_x",
  "sac",
  "sarm"
] as const;

// The base checkpoint each VLA finetunes from, so picking the policy fills the field with a
// model that exists rather than leaving the operator to remember a repo id. Only a default: the
// field stays editable, and a checkpoint trained here is a legitimate value for it too.
const DEFAULT_BASE_CHECKPOINT: Record<string, string> = {
  pi0: "lerobot/pi0_base",
  pi05: "lerobot/pi05_base",
  pi0_fast: "lerobot/pi0fast_base",
  smolvla: "lerobot/smolvla_base"
};

// The workstation's shared model volume contains a verified local pi0.5 checkpoint. Keep this
// host-specific: other training machines must retain the portable Hugging Face repo default.
const TELE_PI05_BASE_CHECKPOINT = "/home/tele/Models/pi05_base";

function defaultBaseCheckpoint(policy: string, machine: TrainingMachine | null): string {
  if (policy === "pi05" && machine?.hostname === "tele-MS-7E07") {
    return TELE_PI05_BASE_CHECKPOINT;
  }
  return DEFAULT_BASE_CHECKPOINT[policy] ?? "";
}

// Policies that are pretrained models to adapt rather than architectures to fit from scratch.
// Training one of these from random weights on a few hundred FR3 episodes is not a smaller
// version of finetuning it -- it is a different, much worse thing, and the page should say so.
const FINETUNE_POLICIES = new Set(Object.keys(DEFAULT_BASE_CHECKPOINT));

// These presets are deliberately specific to the workstation's 30 fps delta-action
// training views. They are not replacements for the policy dataclass defaults used by
// the CLI: an absolute-pose task, a differently sampled dataset, or an established
// rollout latency budget needs its own measured configuration.
const ACT_DELTA_STARTING_POLICY_CONFIG = JSON.stringify({
  chunk_size: 30,
  n_action_steps: 1,
  temporal_ensemble_coeff: 0.01,
  optimizer_lr: 2.5e-5,
  optimizer_lr_backbone: 1e-5
});

const PI05_LORA_24GB_STARTING_POLICY_CONFIG = JSON.stringify({
  dtype: "bfloat16",
  gradient_checkpointing: true,
  compile_model: true,
  optimizer_lr: 1e-4,
  scheduler_warmup_steps: 1000,
  scheduler_decay_steps: 20000,
  scheduler_decay_lr: 1e-5
});

type TrainingPreset = {
  batchSize: string;
  description: string;
  name: string;
  policyConfig: string;
  saveFreq: string;
  steps: string;
};

function startingPreset(policy: string, actionMode: string | undefined): TrainingPreset | null {
  if (policy === "act" && actionMode === "delta_ee_from_prev_cmd") {
    return {
      name: "FR3 delta ACT",
      description:
        "30-frame prediction horizon (1.0 s), re-plan every frame, temporal ensemble 0.01; avoids the upstream 100-frame / 3.33 s open-loop execution.",
      policyConfig: ACT_DELTA_STARTING_POLICY_CONFIG,
      steps: "20000",
      batchSize: "8",
      saveFreq: "2000"
    };
  }
  if (policy === "pi05") {
    return {
      name: "pi0.5 + LoRA on 24 GiB",
      description:
        "BF16, gradient checkpointing, and compilation keep the first adapter run inside a 24 GiB GPU budget; batch size 2 is intentionally conservative.",
      policyConfig: PI05_LORA_24GB_STARTING_POLICY_CONFIG,
      steps: "20000",
      batchSize: "2",
      saveFreq: "2000"
    };
  }
  return null;
}

const RUNNING_STATES = new Set(["syncing", "starting", "running"]);

function gpuLine(gpu: TrainingMachine["gpus"] extends (infer G)[] | undefined ? G : never): string {
  const total = gpu.memoryTotalMb ?? 0;
  const used = gpu.memoryUsedMb ?? 0;
  const free = Math.max(0, total - used);
  return `${(free / 1024).toFixed(1)} / ${(total / 1024).toFixed(1)} GiB free`;
}

/** One past run, as one line in the copy-settings dropdown.
 *
 *  Leads with the date and the knobs that actually distinguish two runs of the same view --
 *  steps, batch, LoRA rank -- because the job names differ only by their trailing timestamp. */
function describeHistoryEntry(entry: TrainingHistoryEntry): string {
  const p = entry.params;
  const day = entry.startedAt ? entry.startedAt.slice(0, 10) : "unknown date";
  const parts = [day, entry.policy || "?", `${(entry.steps || 0).toLocaleString()} steps`];
  if (p.batchSize !== undefined) parts.push(`bs ${p.batchSize}`);
  if (p.loraEnabled) parts.push(`LoRA r${p.loraR ?? "?"}`);
  if (entry.viewName) parts.push(entry.viewName);
  return parts.join(" · ");
}

function describeView(view: TrainingView): string {
  const strides = Object.values(view.frameStride ?? {});
  const resampled = strides.some((stride) => stride > 1);
  const parts = [
    `${view.episodes} ep`,
    `${view.frames.toLocaleString()} frames`,
    `${view.fps} fps${resampled ? " (resampled)" : ""}`,
    view.actionMode || "unknown contract"
  ];
  return parts.join(" · ");
}

export function TrainingPage() {
  const [hosts, setHosts] = useState<TrainingHost[]>([]);
  const [hostId, setHostId] = useState("local");
  const [machine, setMachine] = useState<TrainingMachine | null>(null);
  const [machineLoading, setMachineLoading] = useState(false);
  const [wandb, setWandb] = useState<TrainingWandbStatus | null>(null);
  const [views, setViews] = useState<TrainingView[]>([]);
  const [viewName, setViewName] = useState("");
  const [run, setRun] = useState<TrainingRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const [policy, setPolicy] = useState<string>("act");
  const [jobName, setJobName] = useState("");
  // Whether the operator has typed their own. Until they do, the job name tracks the view and
  // policy -- it names the output directory and the checkpoint the generated inference config
  // points at, so a name left over from a previously selected view would train one view's
  // frames into another view's directory.
  const [jobNameEdited, setJobNameEdited] = useState(false);
  const [steps, setSteps] = useState("20000");
  const [batchSize, setBatchSize] = useState("8");
  const [numWorkers, setNumWorkers] = useState("4");
  const [saveFreq, setSaveFreq] = useState("5000");
  const [logFreq, setLogFreq] = useState("100");
  // Off, like upstream's `PreTrainedConfig.use_amp`. It used to default on here, which put a
  // claim in every checkpoint's train_config.json that the run did not honour: lerobot_train.py
  // never reads use_amp -- it wraps the step in `accelerator.autocast()` and builds its
  // Accelerator without `mixed_precision=`, so the run is fp32 unless accelerate is configured
  // separately. Two checkpoints would have looked like they differed in numeric precision when
  // they did not. See the note under the box.
  const [useAmp, setUseAmp] = useState(false);
  const [policyConfig, setPolicyConfig] = useState("");
  const [policyConfigEdited, setPolicyConfigEdited] = useState(false);

  const [pretrainedPath, setPretrainedPath] = useState("");
  // Whether the operator has typed their own base checkpoint. Until they do, it tracks the
  // policy -- leaving pi0.5's base sitting in the field after switching to ACT would send
  // lerobot_train looking for pi0.5 weights to load into a ResNet.
  const [pretrainedPathEdited, setPretrainedPathEdited] = useState(false);
  const [loraEnabled, setLoraEnabled] = useState(false);
  const [loraR, setLoraR] = useState("16");
  const [loraTargetModules, setLoraTargetModules] = useState("");

  const [wandbEnabled, setWandbEnabled] = useState(false);
  const [wandbProject, setWandbProject] = useState("lerobot");
  const [wandbEntity, setWandbEntity] = useState("");
  const [wandbKeyInput, setWandbKeyInput] = useState("");

  // The dependency install and its log window. `install` is the gateway's status object, so
  // the modal shows a sync started before this page was opened (or reloaded) rather than an
  // empty box next to a machine that is visibly busy.
  // Past runs, newest first, and which one the "copy settings" control is pointing at.
  const [history, setHistory] = useState<TrainingHistoryEntry[]>([]);
  const [historyJob, setHistoryJob] = useState("");

  const [install, setInstall] = useState<DependencyInstall | null>(null);
  const [installOpen, setInstallOpen] = useState(false);

  const [showAddHost, setShowAddHost] = useState(false);
  const [newHostLabel, setNewHostLabel] = useState("");
  const [newHostTarget, setNewHostTarget] = useState("");
  const [newHostDir, setNewHostDir] = useState("");
  const [newHostPython, setNewHostPython] = useState(".venv-fr3/bin/python");

  const logRef = useRef<HTMLPreElement | null>(null);

  const selectedHost = useMemo(() => hosts.find((h) => h.id === hostId), [hosts, hostId]);
  const selectedView = useMemo(() => views.find((v) => v.name === viewName), [views, viewName]);
  const selectedPreset = useMemo(
    () => startingPreset(policy, selectedView?.actionMode),
    [policy, selectedView?.actionMode]
  );
  const isRunning = run !== null && RUNNING_STATES.has(run.state);
  const policySupport = machine?.policies?.[policy];
  const loraSupport = machine?.features?.lora;
  const isFinetunePolicy = FINETUNE_POLICIES.has(policy);
  const installer = machine?.installer;
  // The machine has no uv environment at all. Every policy is blocked by that one fact, so the
  // page says it once, on the machine card, rather than once per policy the operator clicks.
  const needsEnvironment = machine?.ok === true && installer?.willCreateEnvironment === true;
  // Named on the machine card so the environment is a property of the machine on the page the
  // way it is on the box, rather than something the operator discovers one policy at a time.
  const blockedPolicies = useMemo(
    () =>
      Object.entries(machine?.policies ?? {})
        .filter(([, support]) => support.trainable === false)
        .map(([name]) => name),
    [machine]
  );
  const installRunning = install?.state === "running";
  // Everything the selected policy and the ticked options need, asked for in one sync. Two
  // clicks would mean two resolutions of the same environment, and the second one would be
  // deciding what to do about the packages the first had just installed.
  const missingExtras = useMemo(() => {
    const wanted: string[] = [];
    for (const extra of policySupport?.trainable === false ? policySupport.extras ?? [] : []) {
      if (!wanted.includes(extra)) wanted.push(extra);
    }
    if (loraEnabled && loraSupport?.available === false) {
      for (const extra of loraSupport.extras ?? []) {
        if (!wanted.includes(extra)) wanted.push(extra);
      }
    }
    return wanted;
  }, [policySupport, loraSupport, loraEnabled]);

  const refreshHosts = useCallback(async () => {
    const list = await api.fetchTrainingHosts();
    setHosts(list);
    if (!list.some((h) => h.id === hostId) && list.length > 0) {
      setHostId(list[0].id);
    }
  }, [hostId]);

  const refreshMachine = useCallback(async (id: string) => {
    setMachineLoading(true);
    const payload = await api.fetchTrainingMachine(id);
    setMachine(payload?.machine ?? { ok: false, error: "Gateway did not answer the probe." });
    setWandb(payload?.wandb ?? null);
    setMachineLoading(false);
  }, []);

  useEffect(() => {
    void refreshHosts();
    void api.fetchTrainingViews().then(setViews);
    void api.fetchTrainingHistory().then(setHistory);
    // Once, on mount: an install started from another browser tab -- or before this page was
    // reloaded -- is still running on the machine, and the poll below only starts once
    // something says one is.
    void api.fetchDependencyInstall().then(setInstall);
  }, [refreshHosts]);

  useEffect(() => {
    if (hostId) void refreshMachine(hostId);
  }, [hostId, refreshMachine]);

  // Polled rather than pushed into the snapshot: a training run outlives most page visits,
  // and the operator's question while it runs is only "what step is it on".
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      const status = await api.fetchTrainingStatus();
      if (!cancelled) setRun(status);
    };
    void tick();
    const timer = window.setInterval(tick, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [run?.lastLines?.length]);

  // Re-scan the artifact list when a run stops running. Checkpoints appear on disk while
  // training is still going (every save_freq steps), but the moment the operator cares about
  // is the one where the run is done and there is something to roll out.
  const [artifactToken, setArtifactToken] = useState(0);
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !isRunning) setArtifactToken((value) => value + 1);
    wasRunning.current = isRunning;
  }, [isRunning]);

  useEffect(() => {
    if (selectedView && !jobNameEdited) setJobName(`${selectedView.name}__${policy}`);
  }, [selectedView, policy, jobNameEdited]);

  useEffect(() => {
    if (!pretrainedPathEdited) setPretrainedPath(defaultBaseCheckpoint(policy, machine));
  }, [policy, machine, pretrainedPathEdited]);

  // Seed a safe first run after the QC-gated view has arrived. Once the operator edits the
  // JSON it remains theirs; the button below is the explicit way to return to this preset.
  useEffect(() => {
    if (!policyConfigEdited && selectedPreset) setPolicyConfig(selectedPreset.policyConfig);
  }, [policyConfigEdited, selectedPreset]);

  // LoRA has nothing to adapt without a base model, and the start button would otherwise be
  // enabled for a run the gateway is about to refuse.
  useEffect(() => {
    if (!pretrainedPath.trim()) setLoraEnabled(false);
  }, [pretrainedPath]);

  // Polled only while there is something to watch: an install lasts minutes, and the rest of
  // the time this would be a request per second asking a machine that is not installing
  // anything whether it has finished not installing it.
  useEffect(() => {
    if (!installOpen && !installRunning) return;
    let cancelled = false;
    const tick = async () => {
      const status = await api.fetchDependencyInstall();
      if (!cancelled) setInstall(status);
    };
    void tick();
    const timer = window.setInterval(tick, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [installOpen, installRunning]);

  // Re-probe once, when the install stops running. The whole point of the button is the
  // machine's answer changing, and an operator should not have to know to press Refresh to
  // see it.
  const wasInstalling = useRef(false);
  useEffect(() => {
    if (wasInstalling.current && !installRunning) void refreshMachine(hostId);
    wasInstalling.current = installRunning;
  }, [installRunning, hostId, refreshMachine]);

  const wrap = async (label: string, action: () => Promise<{ ok: boolean; error?: string }>) => {
    setBusy(true);
    setError("");
    setNotice("");
    const result = await action();
    setBusy(false);
    if (!result.ok) setError(result.error || `${label} failed.`);
    return result;
  };

  const onAddHost = async () => {
    const result = await wrap("Add host", () =>
      api.addTrainingHost({
        label: newHostLabel,
        sshTarget: newHostTarget,
        repoDir: newHostDir,
        pythonPath: newHostPython
      })
    );
    if (result.ok) {
      setShowAddHost(false);
      setNewHostLabel("");
      setNewHostTarget("");
      setNewHostDir("");
      await refreshHosts();
      setNotice("Training host added.");
    }
  };

  const onInstallDeps = async (extras: string[]) => {
    setInstallOpen(true);
    const result = await wrap("Install dependencies", () =>
      api.installTrainingDeps(hostId, extras)
    );
    if (result.ok) setInstall((result as { install?: DependencyInstall }).install ?? null);
  };

  const onSync = async () => {
    const result = await wrap("Sync", () => api.syncTrainingHost(hostId));
    if (result.ok) {
      const sync = (result as { sync?: { message?: string } }).sync;
      setNotice(sync?.message || "Sync complete.");
      await refreshMachine(hostId);
    }
  };

  const onSaveWandbKey = async () => {
    const result = await wrap("Save W&B key", () => api.setTrainingWandbKey(hostId, wandbKeyInput));
    if (result.ok) {
      setWandbKeyInput("");
      setWandb((result as { wandb?: TrainingWandbStatus }).wandb ?? null);
      setNotice("W&B API key stored for this machine.");
    }
  };

  const onClearWandbKey = async () => {
    const result = await wrap("Clear W&B key", () => api.clearTrainingWandbKey(hostId));
    if (result.ok) {
      setWandb((result as { wandb?: TrainingWandbStatus }).wandb ?? null);
      setNotice("W&B API key removed.");
    }
  };

  const onStart = async () => {
    const result = await wrap("Start training", () =>
      api.startTraining({
        hostId,
        viewName,
        policy,
        jobName,
        steps: Number(steps) || 20000,
        batchSize: Number(batchSize) || 8,
        numWorkers: Number(numWorkers) || 4,
        saveFreq: Number(saveFreq) || 5000,
        logFreq: Number(logFreq) || 100,
        device: "auto",
        useAmp,
        policyConfig,
        pretrainedPath: pretrainedPath.trim(),
        loraEnabled,
        loraR: Number(loraR) || 16,
        loraTargetModules,
        wandbEnabled,
        wandbProject,
        wandbEntity
      })
    );
    if (result.ok) {
      setRun((result as { training?: TrainingRun }).training ?? null);
      setNotice("Training started.");
    }
  };

  const applyStartingPreset = (preset: TrainingPreset) => {
    setSteps(preset.steps);
    setBatchSize(preset.batchSize);
    setSaveFreq(preset.saveFreq);
    setPolicyConfig(preset.policyConfig);
    setPolicyConfigEdited(false);
    setNotice(`${preset.name} starting preset applied.`);
  };

  /** Refill the form from a past run.
   *
   *  The view is deliberately not copied: the reason to reach for an earlier run's settings is
   *  almost always to put them on frames that did not exist then. The job name is not copied
   *  either -- it is derived per start, and reusing one would train into a directory that is
   *  already another run's record.
   *
   *  A key the recorded run did not carry leaves that field alone rather than resetting it to a
   *  default the old run never actually used.
   */
  const applyHistory = (entry: TrainingHistoryEntry) => {
    const p = entry.params;
    if (p.policy !== undefined) setPolicy(p.policy);
    if (p.steps !== undefined) setSteps(String(p.steps));
    if (p.batchSize !== undefined) setBatchSize(String(p.batchSize));
    if (p.numWorkers !== undefined) setNumWorkers(String(p.numWorkers));
    if (p.saveFreq !== undefined) setSaveFreq(String(p.saveFreq));
    if (p.logFreq !== undefined) setLogFreq(String(p.logFreq));
    if (p.useAmp !== undefined) setUseAmp(p.useAmp);
    if (p.policyConfig !== undefined) {
      setPolicyConfig(p.policyConfig);
      // Marked as hand-edited so selecting a policy afterwards does not quietly overwrite the
      // config this run is being reproduced from with that policy's starting preset.
      setPolicyConfigEdited(p.policyConfig.trim() !== "");
    }
    if (p.pretrainedPath !== undefined) {
      setPretrainedPath(p.pretrainedPath);
      setPretrainedPathEdited(p.pretrainedPath.trim() !== "");
    }
    if (p.loraEnabled !== undefined) setLoraEnabled(p.loraEnabled);
    if (p.loraR !== undefined) setLoraR(String(p.loraR));
    if (p.loraTargetModules !== undefined) setLoraTargetModules(p.loraTargetModules);
    if (p.wandbEnabled !== undefined) setWandbEnabled(p.wandbEnabled);
    if (p.wandbProject !== undefined) setWandbProject(p.wandbProject);
    if (p.wandbEntity !== undefined) setWandbEntity(p.wandbEntity);
    setNotice(
      `Settings copied from ${entry.jobName}. The training view and job name were left alone.`
    );
  };

  const onPolicyChange = (nextPolicy: string) => {
    setPolicy(nextPolicy);
    // A config copied from ACT into pi0.5 (or vice versa) is worse than no config. Selecting
    // a policy therefore seeds its applicable preset rather than retaining stale JSON.
    setPolicyConfigEdited(false);
    const preset = startingPreset(nextPolicy, selectedView?.actionMode);
    if (preset) applyStartingPreset(preset);
    else setPolicyConfig("");
  };

  const onStop = async () => {
    const result = await wrap("Stop training", () => api.stopTraining());
    if (result.ok) setNotice("Stop signal sent.");
  };

  const startDisabled =
    busy || isRunning || !viewName || !jobName || (selectedView?.episodes ?? 0) < 1;

  return (
    <div className="page">
      <PageHeader
        title="Training"
        subtitle="Train a policy on a QC-gated training view, here or on another machine."
      />

      {error && <div className="banner banner-error">{error}</div>}
      {notice && !error && <div className="banner banner-ok">{notice}</div>}

      {/* ------------------------------------------------------------ machine --- */}
      <section className="card">
        <div className="card-head">
          <h3>Training machine</h3>
          <div className="row-actions">
            <button type="button" onClick={() => void refreshMachine(hostId)} disabled={machineLoading}>
              {machineLoading ? "Probing…" : "Refresh"}
            </button>
            <button type="button" onClick={() => setShowAddHost((open) => !open)}>
              {showAddHost ? "Cancel" : "Add remote…"}
            </button>
          </div>
        </div>

        <label className="field">
          <span>Machine</span>
          <select value={hostId} onChange={(event) => setHostId(event.target.value)} disabled={isRunning}>
            {hosts.map((host) => (
              <option key={host.id} value={host.id}>
                {host.label}
                {host.kind === "remote" ? ` — ${host.sshTarget}:${host.repoDir}` : ""}
              </option>
            ))}
          </select>
        </label>

        {showAddHost && (
          <div className="subcard">
            <p className="hint">
              A remote machine trains from <em>its own</em> checkout, so starting a run there syncs
              this repo across first. Login must be by ssh key — the gateway never prompts for a
              password.
            </p>
            <label className="field">
              <span>Label</span>
              <input value={newHostLabel} onChange={(e) => setNewHostLabel(e.target.value)} placeholder="Training box A" />
            </label>
            <label className="field">
              <span>SSH target</span>
              <input value={newHostTarget} onChange={(e) => setNewHostTarget(e.target.value)} placeholder="user@192.168.1.50" />
            </label>
            <label className="field">
              <span>Repo directory</span>
              <input value={newHostDir} onChange={(e) => setNewHostDir(e.target.value)} placeholder="/home/user/lerobot" />
            </label>
            <label className="field">
              <span>Python</span>
              <input value={newHostPython} onChange={(e) => setNewHostPython(e.target.value)} />
            </label>
            <button type="button" onClick={() => void onAddHost()} disabled={busy}>
              Add machine
            </button>
          </div>
        )}

        {machine && !machine.ok && (
          <div className="banner banner-error">
            Probe failed: {machine.error}
            {machine.detail?.length ? <pre className="log-block">{machine.detail.join("\n")}</pre> : null}
          </div>
        )}

        {machine?.ok && (
          <>
            <div className="metric-row">
              <Metric label="Host" value={machine.hostname ?? "—"} />
              <Metric label="CPU threads" value={machine.cpuCount ?? "—"} />
              <Metric label="Python" value={machine.python?.version ?? "—"} />
              <Metric
                label="Torch"
                value={machine.torch?.installed ? (machine.torch.version ?? "yes") : "not installed"}
              />
              <Metric
                label="CUDA"
                value={
                  machine.torch?.cudaAvailable
                    ? `${machine.torch.cudaVersion ?? "?"} · ${machine.torch.deviceCount ?? 0} dev`
                    : "unavailable"
                }
              />
              <Metric
                label="Disk free"
                value={machine.disk?.freeGb != null ? `${machine.disk.freeGb} GB` : "—"}
              />
            </div>

            {machine.gpus?.length ? (
              <table className="table">
                <thead>
                  <tr>
                    <th>GPU</th>
                    <th>Memory</th>
                    <th>Util</th>
                    <th>Temp</th>
                    <th>Driver</th>
                  </tr>
                </thead>
                <tbody>
                  {machine.gpus.map((gpu) => (
                    <tr key={gpu.index ?? gpu.name}>
                      <td>
                        #{gpu.index} {gpu.name}
                      </td>
                      <td>{gpuLine(gpu)}</td>
                      <td>{gpu.utilizationPct != null ? `${gpu.utilizationPct}%` : "—"}</td>
                      <td>{gpu.temperatureC != null ? `${gpu.temperatureC}°C` : "—"}</td>
                      <td>{gpu.driverVersion}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="hint">No GPU reported{machine.gpuError ? ` — ${machine.gpuError}` : "."}</p>
            )}

            {needsEnvironment ? (
              <div className="banner banner-warn">
                <div>
                  {selectedHost?.label} has no <code>{installer?.venvPath ?? ".venv-fr3"}</code>{" "}
                  yet, so nothing can train there. Dependencies here are managed with uv; this
                  builds the environment and installs{" "}
                  {missingExtras.length > 0
                    ? `${missingExtras.join(" + ")} along with everything else that machine needs.`
                    : "everything that machine needs."}
                </div>
                <DependencyInstallControls
                  extras={missingExtras}
                  installer={installer}
                  hostLabel={selectedHost?.label ?? ""}
                  busy={busy}
                  installRunning={installRunning}
                  trainingRunning={isRunning}
                  onInstall={() => void onInstallDeps(missingExtras)}
                  onShowLog={() => setInstallOpen(true)}
                  hasLog={Boolean(install && install.state !== "idle")}
                />
              </div>
            ) : (
              blockedPolicies.length > 0 && (
                <p className="hint">
                  Cannot train here yet: <strong>{blockedPolicies.join(", ")}</strong>. Pick one
                  below to see what it needs and install it.
                  {installer && !installer.canInstall ? ` ${installer.reason}` : ""}
                </p>
              )
            )}

            {selectedHost?.kind === "remote" && (
              <div className="row-actions">
                <button type="button" onClick={() => void onSync()} disabled={busy || isRunning}>
                  Sync code now
                </button>
                <span className="hint">
                  {machine.repoRootExists
                    ? `Repo present at ${machine.repoRoot}`
                    : `No repo at ${machine.repoRoot} — sync creates it.`}
                </span>
              </div>
            )}
          </>
        )}
      </section>

      {/* --------------------------------------------------------------- wandb --- */}
      <section className="card">
        <div className="card-head">
          <h3>Weights &amp; Biases</h3>
          <StatusDot state={wandb?.configured ? "ok" : "idle"} />
        </div>
        <p className="hint">
          The key is stored per machine outside the repo, mode 0600, and is never sent back to
          this page or written into a command line. For a remote machine it is copied over ssh
          into a 0600 file on that machine at run time.
        </p>
        {wandb?.configured ? (
          <div className="row-actions">
            <span>
              Key stored for this machine (…{wandb.keySuffix})
            </span>
            <button type="button" onClick={() => void onClearWandbKey()} disabled={busy}>
              Remove key
            </button>
          </div>
        ) : (
          <div className="row-actions">
            <input
              type="password"
              value={wandbKeyInput}
              onChange={(event) => setWandbKeyInput(event.target.value)}
              placeholder="W&B API key"
              autoComplete="off"
            />
            <button type="button" onClick={() => void onSaveWandbKey()} disabled={busy || !wandbKeyInput}>
              Save key
            </button>
          </div>
        )}
        <label className="field-inline">
          <input
            type="checkbox"
            checked={wandbEnabled}
            onChange={(event) => setWandbEnabled(event.target.checked)}
            disabled={!wandb?.configured || isRunning}
          />
          <span>Log this run to W&amp;B</span>
        </label>
        {wandbEnabled && (
          <div className="field-row">
            <label className="field">
              <span>Project</span>
              <input value={wandbProject} onChange={(e) => setWandbProject(e.target.value)} />
            </label>
            <label className="field">
              <span>Entity</span>
              <input value={wandbEntity} onChange={(e) => setWandbEntity(e.target.value)} placeholder="(default)" />
            </label>
          </div>
        )}
      </section>

      {/* ---------------------------------------------------------------- run --- */}
      <section className="card">
        <div className="card-head">
          <h3>Run</h3>
          <StatusDot state={run?.state ?? "idle"} />
        </div>

        <label className="field">
          <span>Training view</span>
          <select
            value={viewName}
            onChange={(event) => setViewName(event.target.value)}
            disabled={isRunning}
          >
            <option value="">Select a training view…</option>
            {views.map((view) => (
              <option key={view.name} value={view.name}>
                {view.name} — {describeView(view)}
              </option>
            ))}
          </select>
        </label>
        {selectedView && (
          <p className="hint">
            {selectedView.cameras.map((cam) => cam.replace("observation.images.", "")).join(" + ")} ·{" "}
            {describeView(selectedView)}
            {selectedView.episodes < 2 && (
              <strong> — only {selectedView.episodes} episode; too little to train anything real.</strong>
            )}
          </p>
        )}

        <label className="field">
          <span>Copy settings from a past run</span>
          <select
            value={historyJob}
            onChange={(event) => {
              const jobName = event.target.value;
              setHistoryJob(jobName);
              const entry = history.find((item) => item.jobName === jobName);
              if (entry) applyHistory(entry);
            }}
            disabled={isRunning || history.length === 0}
          >
            <option value="">
              {history.length === 0
                ? "No past runs found under outputs/train"
                : "Start from the current settings…"}
            </option>
            {history.map((entry) => (
              <option key={entry.jobName} value={entry.jobName}>
                {describeHistoryEntry(entry)}
              </option>
            ))}
          </select>
        </label>

        <div className="field-row">
          <label className="field">
            <span>Policy</span>
            <select value={policy} onChange={(event) => onPolicyChange(event.target.value)} disabled={isRunning}>
              <optgroup label="Trained and rolled out on this rig">
                {VERIFIED_POLICIES.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </optgroup>
              <optgroup label="Never run here — check the config yourself">
                {OTHER_POLICIES.map((name) => (
                  <option key={name} value={name}>
                    {name}
                    {machine?.policies?.[name]?.trainable === false ? " (deps missing)" : ""}
                  </option>
                ))}
              </optgroup>
            </select>
          </label>
          <label className="field">
            <span>Job name</span>
            <input
              value={jobName}
              onChange={(event) => {
                setJobNameEdited(event.target.value.trim() !== "");
                setJobName(event.target.value);
              }}
              disabled={isRunning}
            />
          </label>
        </div>
        <p className="hint">
          Starting appends a UTC timestamp to this name, so each run owns{" "}
          <code>outputs/train/{jobName || "<job>"}__&lt;started-at&gt;</code> and retraining the
          same view neither collides with the previous run nor overwrites its checkpoints. The
          run&apos;s log file carries the same stamp.
        </p>

        {selectedPreset && (
          <div className="banner banner-warn">
            <div>
              <strong>{selectedPreset.name} starting preset.</strong> {selectedPreset.description}
            </div>
            <div className="row-actions">
              <button type="button" onClick={() => applyStartingPreset(selectedPreset)} disabled={isRunning}>
                Apply starting preset
              </button>
            </div>
          </div>
        )}

        {policySupport && !policySupport.trainable && (
          <div className="banner banner-warn">
            <div>
              {policy} cannot run on {selectedHost?.label}: missing{" "}
              {policySupport.missing.join(", ")}.
              {missingExtras.length > 0
                ? ` Installing ${missingExtras.join(" + ")} on that machine fixes it.`
                : " Those are base dependencies of this project, so syncing the environment" +
                  " is the whole fix."}
            </div>
            {needsEnvironment ? (
              <div className="hint">
                Build the environment first — the button on the machine card above does it, and
                the plan already includes what {policy} needs.
              </div>
            ) : (
              <DependencyInstallControls
                extras={missingExtras}
                installer={installer}
                hostLabel={selectedHost?.label ?? ""}
                busy={busy}
                installRunning={installRunning}
                trainingRunning={isRunning}
                onInstall={() => void onInstallDeps(missingExtras)}
                onShowLog={() => setInstallOpen(true)}
                hasLog={Boolean(install && install.state !== "idle")}
              />
            )}
          </div>
        )}

        <div className="field-row">
          <label className="field">
            <span>Steps</span>
            <input value={steps} onChange={(e) => setSteps(e.target.value)} disabled={isRunning} />
          </label>
          <label className="field">
            <span>Batch size</span>
            <input value={batchSize} onChange={(e) => setBatchSize(e.target.value)} disabled={isRunning} />
          </label>
          <label className="field">
            <span>Workers</span>
            <input value={numWorkers} onChange={(e) => setNumWorkers(e.target.value)} disabled={isRunning} />
          </label>
          <label className="field">
            <span>Save every</span>
            <input value={saveFreq} onChange={(e) => setSaveFreq(e.target.value)} disabled={isRunning} />
          </label>
          <label className="field">
            <span>Log every</span>
            <input value={logFreq} onChange={(e) => setLogFreq(e.target.value)} disabled={isRunning} />
          </label>
        </div>

        <div className="field-row">
          <label className="field">
            <span>Base checkpoint (optional)</span>
            <input
              value={pretrainedPath}
              onChange={(event) => {
                setPretrainedPathEdited(true);
                setPretrainedPath(event.target.value);
              }}
              placeholder="lerobot/pi05_base, or a local checkpoint directory"
              disabled={isRunning}
            />
          </label>
          <label className="field-inline">
            <input
              type="checkbox"
              checked={loraEnabled}
              onChange={(event) => setLoraEnabled(event.target.checked)}
              disabled={isRunning || !pretrainedPath.trim() || loraSupport?.available === false}
            />
            <span>LoRA (freeze the base, train an adapter)</span>
          </label>
        </div>
        {loraEnabled && (
          <div className="field-row">
            <label className="field">
              <span>LoRA rank</span>
              <input value={loraR} onChange={(e) => setLoraR(e.target.value)} disabled={isRunning} />
            </label>
            <label className="field">
              <span>Target modules (optional)</span>
              <input
                value={loraTargetModules}
                onChange={(e) => setLoraTargetModules(e.target.value)}
                placeholder="(policy default)"
                disabled={isRunning}
              />
            </label>
          </div>
        )}
        <p className="hint">
          A base checkpoint supplies <em>weights only</em> — every hyperparameter still comes from
          this page. Left empty, {policy} trains from random initialization.{" "}
          {isFinetunePolicy && !pretrainedPath.trim() && (
            <strong>
              {policy} is a pretrained model meant to be adapted; from scratch on one rig&apos;s
              episodes it will not learn the task.
            </strong>
          )}{" "}
          {loraEnabled ? (
            <>
              LoRA trains an adapter on frozen base weights — far less GPU memory, and the
              checkpoint holds the adapter rather than the whole model. Leave{" "}
              <em>Target modules</em> empty to use the policy&apos;s own default: pi0.5 targets its
              action expert&apos;s q/v projections plus the state and action projections, which is
              the tuned answer for it. Rank and targets are the only adapter knobs LeRobot&apos;s{" "}
              <code>PeftConfig</code> exposes — there is no alpha or dropout field to set.
            </>
          ) : null}
        </p>
        {loraSupport?.available === false && (
          <div className="banner banner-warn">
            <div>
              LoRA is unavailable on {selectedHost?.label}: missing{" "}
              {loraSupport.missing.join(", ")}.
            </div>
            {!needsEnvironment && (
              <DependencyInstallControls
                extras={loraSupport.extras ?? []}
                installer={installer}
                hostLabel={selectedHost?.label ?? ""}
                busy={busy}
                installRunning={installRunning}
                trainingRunning={isRunning}
                onInstall={() => void onInstallDeps(loraSupport.extras ?? [])}
                onShowLog={() => setInstallOpen(true)}
                hasLog={Boolean(install && install.state !== "idle")}
              />
            )}
          </div>
        )}
        {isFinetunePolicy && (
          <p className="hint">
            First run only: pi0/pi0.5 tokenize their prompt with{" "}
            <code>google/paligemma-3b-pt-224</code>, a <strong>gated</strong> Hugging Face repo.
            Accept its terms on the model page and <code>hf auth login</code> on the training
            machine, or the run stops in the pre-processor with a 401.
          </p>
        )}

        <label className="field">
          <span>Policy config (JSON, optional)</span>
          <input
            value={policyConfig}
            onChange={(event) => {
              setPolicyConfigEdited(true);
              setPolicyConfig(event.target.value);
            }}
            placeholder='{"chunk_size": 50, "optimizer_lr": 2.5e-5}'
            disabled={isRunning}
          />
        </label>
        <p className="hint">
          {selectedPreset
            ? `The ${selectedPreset.name} JSON above is the selected starting point. Anything typed here wins over it.`
            : `Left empty, ${policy} trains on its own upstream LeRobot defaults (ACT: chunk_size 100, n_action_steps 100, lr 1e-5). Anything typed here wins over them.`}
        </p>

        <label className="field-inline">
          <input
            type="checkbox"
            checked={useAmp}
            onChange={(event) => setUseAmp(event.target.checked)}
            disabled={isRunning}
          />
          <span>Mixed precision (AMP)</span>
        </label>
        <p className="hint">
          Recorded into the checkpoint&apos;s config, but not acted on:{" "}
          <code>lerobot_train.py</code> leaves precision to Accelerate and never passes it{" "}
          <code>mixed_precision</code>, so training runs fp32 either way. Ticking this changes what
          the run says about itself, not what it does.
        </p>

        <div className="row-actions">
          <button type="button" onClick={() => void onStart()} disabled={startDisabled}>
            {selectedHost?.kind === "remote" ? "Sync & start training" : "Start training"}
          </button>
          <button type="button" onClick={() => void onStop()} disabled={busy || !isRunning}>
            Stop
          </button>
        </div>
      </section>

      {/* -------------------------------------------------------------- status --- */}
      {run && run.state !== "idle" && (
        <section className="card">
          <div className="card-head">
            <h3>
              {run.jobName || "Training"} — {run.state}
            </h3>
            <StatusDot state={run.state} />
          </div>
          <div className="metric-row">
            <Metric label="Machine" value={run.hostLabel || run.hostId} />
            <Metric label="Policy" value={run.policy} />
            <Metric label="View" value={run.viewName} />
            <Metric
              label="Step"
              value={run.totalSteps ? `${run.step} / ${run.totalSteps}` : String(run.step)}
            />
            <Metric label="Loss" value={run.loss != null ? run.loss.toFixed(5) : "—"} />
            <Metric label="Output" value={run.outputDir} />
          </div>
          {run.wandbUrl && (
            <p className="hint">
              W&amp;B run:{" "}
              <a href={run.wandbUrl} target="_blank" rel="noreferrer">
                {run.wandbUrl}
              </a>
            </p>
          )}
          <p className="hint">Log file: {run.logPath}</p>
          <pre className="log-block" ref={logRef}>
            {(run.lastLines ?? []).join("\n") || run.message}
          </pre>
        </section>
      )}

      {/* ----------------------------------------------------------- artifacts --- */}
      <section className="card">
        <div className="card-head">
          <h3>Checkpoints</h3>
        </div>
        <p className="hint">
          What training produced, on this machine and on every training host. A checkpoint
          trained elsewhere has to be fetched here before it can be rolled out — the robot and
          its cameras are on this machine. Roll one out on the{" "}
          <a href="#/rollout">Rollout</a> page.
        </p>
        <CheckpointBrowser mode="manage" refreshToken={artifactToken} />
      </section>

      {installOpen && (
        <Modal
          title="Training dependencies"
          className="cali-modal-wide"
          onClose={() => setInstallOpen(false)}
          footer={
            <>
              <span className="hint">
                {installRunning
                  ? "Closing this leaves the install running; reopen it from the banner."
                  : install?.logPath || ""}
              </span>
              <button type="button" onClick={() => setInstallOpen(false)}>
                {installRunning ? "Hide" : "Close"}
              </button>
            </>
          }
        >
          <InstallLogView install={install} />
        </Modal>
      )}
    </div>
  );
}

/** The Install button, plus the reasons it is not one. */
function DependencyInstallControls({
  extras,
  installer,
  hostLabel,
  busy,
  installRunning,
  trainingRunning,
  onInstall,
  onShowLog,
  hasLog
}: {
  extras: string[];
  installer: TrainingMachine["installer"];
  hostLabel: string;
  busy: boolean;
  installRunning: boolean;
  trainingRunning: boolean;
  onInstall: () => void;
  onShowLog: () => void;
  hasLog: boolean;
}) {
  // Every one of these is a state the operator can be in, and each has a different answer.
  // Collapsing them into a disabled button with no explanation is how a page ends up being
  // described as broken.
  const blocked = trainingRunning
    ? "A training run is using this environment; stop it first."
    : installRunning
      ? "An install is already running."
      : installer && !installer.canInstall
        ? installer.reason
        : "";

  // No extra to name is not nothing to do: torch and accelerate are base dependencies, so a
  // bare `uv sync` is the fix, and it is also what a machine with no environment at all needs.
  const building = installer?.willCreateEnvironment === true;
  const label = building
    ? "Build environment with uv"
    : extras.length > 0
      ? `Install ${extras.join(" + ")}`
      : "Sync base dependencies";

  return (
    <div className="row-actions">
      <button type="button" onClick={onInstall} disabled={busy || Boolean(blocked)}>
        {installRunning ? "Installing…" : label}
      </button>
      {hasLog && (
        <button type="button" onClick={onShowLog}>
          View install log
        </button>
      )}
      <span className="hint">
        {blocked ||
          (building
            ? `${hostLabel} has no ${installer?.venvPath ?? ".venv-fr3"} yet. This builds it with ` +
              "uv, including everything else that machine needs — several GB, once."
            : `Runs uv sync on ${hostLabel} and streams the log here. It adds packages; it ` +
              "never removes one, so the recorder's own dependencies survive it.")}
      </span>
    </div>
  );
}

/** The install's own output, tailed. */
function InstallLogView({ install }: { install: DependencyInstall | null }) {
  const ref = useRef<HTMLPreElement | null>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [install?.lastLines?.length]);

  if (!install || install.state === "idle") {
    return <p className="hint">Nothing has been installed from this page yet.</p>;
  }
  return (
    <>
      <div className="metric-row">
        <Metric label="Machine" value={install.hostLabel || install.hostId} />
        <Metric label="Extras" value={install.extras.join(", ")} />
        <Metric label="State" value={install.state} />
      </div>
      <p className="hint">{install.message}</p>
      <p className="hint">
        <code>{install.command}</code>
      </p>
      <pre className="log-block" ref={ref}>
        {install.lastLines.join("\n") || "Waiting for output…"}
      </pre>
      {install.state === "complete" && (
        <p className="hint">
          The machine has been re-probed; the banner above now reflects what it reports.
        </p>
      )}
    </>
  );
}
