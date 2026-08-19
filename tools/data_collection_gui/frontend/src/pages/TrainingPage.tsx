import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../apiClient";
import type {
  TrainingHost,
  TrainingMachine,
  TrainingRun,
  TrainingView,
  TrainingWandbStatus
} from "../types";
import { Metric, PageHeader, StatusDot } from "../shared/ui";

// Mirrors KNOWN_POLICY_TYPES in tools/fr3/fr3_train_il_policy.py. Split by what this repo
// has actually tuned for the FR3 rig, because "selectable" and "has defaults worth using"
// are different claims and the page should not blur them.
const TUNED_POLICIES = ["act", "diffusion"] as const;
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

const RUNNING_STATES = new Set(["syncing", "starting", "running"]);

function gpuLine(gpu: TrainingMachine["gpus"] extends (infer G)[] | undefined ? G : never): string {
  const total = gpu.memoryTotalMb ?? 0;
  const used = gpu.memoryUsedMb ?? 0;
  const free = Math.max(0, total - used);
  return `${(free / 1024).toFixed(1)} / ${(total / 1024).toFixed(1)} GiB free`;
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
  const [useAmp, setUseAmp] = useState(true);
  const [policyConfig, setPolicyConfig] = useState("");

  const [wandbEnabled, setWandbEnabled] = useState(false);
  const [wandbProject, setWandbProject] = useState("lerobot");
  const [wandbEntity, setWandbEntity] = useState("");
  const [wandbKeyInput, setWandbKeyInput] = useState("");

  const [showAddHost, setShowAddHost] = useState(false);
  const [newHostLabel, setNewHostLabel] = useState("");
  const [newHostTarget, setNewHostTarget] = useState("");
  const [newHostDir, setNewHostDir] = useState("");
  const [newHostPython, setNewHostPython] = useState(".venv-fr3/bin/python");

  const logRef = useRef<HTMLPreElement | null>(null);

  const selectedHost = useMemo(() => hosts.find((h) => h.id === hostId), [hosts, hostId]);
  const selectedView = useMemo(() => views.find((v) => v.name === viewName), [views, viewName]);
  const isRunning = run !== null && RUNNING_STATES.has(run.state);
  const policySupport = machine?.policies?.[policy];

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

  useEffect(() => {
    if (selectedView && !jobNameEdited) setJobName(`${selectedView.name}__${policy}`);
  }, [selectedView, policy, jobNameEdited]);

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

        <div className="field-row">
          <label className="field">
            <span>Policy</span>
            <select value={policy} onChange={(event) => setPolicy(event.target.value)} disabled={isRunning}>
              <optgroup label="Tuned for this rig">
                {TUNED_POLICIES.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </optgroup>
              <optgroup label="Policy defaults apply — set --policy-config yourself">
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

        {policySupport && !policySupport.trainable && (
          <div className="banner banner-warn">
            {policy} cannot run on {selectedHost?.label}: missing {policySupport.missing.join(", ")}.
            Install the matching extra there before starting.
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

        <label className="field">
          <span>Policy config (JSON, optional)</span>
          <input
            value={policyConfig}
            onChange={(event) => setPolicyConfig(event.target.value)}
            placeholder='{"chunk_size": 50, "optimizer_lr": 2.5e-5}'
            disabled={isRunning}
          />
        </label>

        <label className="field-inline">
          <input
            type="checkbox"
            checked={useAmp}
            onChange={(event) => setUseAmp(event.target.checked)}
            disabled={isRunning}
          />
          <span>Mixed precision (AMP)</span>
        </label>

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
    </div>
  );
}
