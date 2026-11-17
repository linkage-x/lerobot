import { useCallback, useEffect, useMemo, useState } from "react";
import { PageHeader, userGuideUrl } from "../shared/ui";
import {
  type Checkpoint,
  type ExportCheck,
  type ProbeResult,
  type TrainingExport,
  type TrainingRun,
  type TrainingStatus,
  trainingApi
} from "./trainingApi";
import {
  type Crop,
  type Policy,
  type TrainingForm,
  DEFAULT_FORM,
  chunkSeconds,
  formProblems,
  outputSize,
  policyDefaults,
  progressFraction,
  startRequest,
  stepHz,
  suggestHoldout
} from "./trainingForm";
import { TrainingMachines } from "./TrainingMachines";
import { gpusFor } from "./trainingGpus";

const ACTIVE = new Set(["syncing", "starting", "running"]);

function NumberField({
  label,
  value,
  onChange,
  hint
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  hint?: string;
}) {
  return (
    <label className="train-field">
      <span>{label}</span>
      <input type="number" value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {hint ? <small>{hint}</small> : null}
    </label>
  );
}

function CameraEditor({
  exportName,
  hand,
  name,
  crop,
  size,
  longSide,
  removable,
  onCrop,
  onRemove
}: {
  exportName: string;
  hand: string;
  name: string;
  crop: Crop | null;
  size: [number, number] | undefined;
  longSide: number;
  removable: boolean;
  onCrop: (crop: Crop | null) => void;
  onRemove: () => void;
}) {
  const [at, setAt] = useState(0.5);
  const [note, setNote] = useState("");
  const full: Crop | null = size ? [0, 0, size[0], size[1]] : null;
  const box = crop ?? full;
  const out = box ? outputSize(box, longSide) : null;
  const suggest = async () => {
    setNote("Looking at where the gripper rig was seen…");
    try {
      const result = await trainingApi.crop(exportName, name, hand);
      onCrop(result.crop);
      setNote(result.note);
    } catch (error) {
      setNote(String(error instanceof Error ? error.message : error));
    }
  };
  return (
    <div className="train-camera">
      <div className="train-camera-head">
        <strong>{name}</strong>
        <span>
          {box ? `crop ${box.join(", ")}` : "no size"} {out ? `→ ${out[0]}×${out[1]}` : ""}
        </span>
        {name !== "cam_10" ? <button onClick={suggest}>Suggest crop</button> : null}
        {crop ? <button onClick={() => onCrop(null)}>Whole frame</button> : null}
        {removable ? <button onClick={onRemove}>Remove</button> : null}
      </div>
      {box ? (
        <div className="train-crop-inputs">
          {(["x0", "y0", "x1", "y1"] as const).map((label, i) => (
            <label key={label}>
              {label}
              <input
                type="number"
                value={box[i]}
                onChange={(e) => {
                  const next = [...box] as Crop;
                  next[i] = Number(e.target.value);
                  onCrop(next);
                }}
              />
            </label>
          ))}
          <label>
            frame
            <input type="range" min={0} max={0.99} step={0.01} value={at} onChange={(e) => setAt(Number(e.target.value))} />
          </label>
        </div>
      ) : null}
      {note ? <p className="panel-note">{note}</p> : null}
      {exportName ? (
        <img className="train-preview" alt={`${name} with its crop`} src={trainingApi.frameUrl(exportName, name, crop, at)} />
      ) : null}
    </div>
  );
}

function RunDetail({ run, onAction }: { run: TrainingRun; onAction: (fn: () => Promise<unknown>) => void }) {
  const [log, setLog] = useState<string[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [moreSteps, setMoreSteps] = useState<number>(run.totalSteps * 2);
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const [lines, ckpts] = await Promise.all([trainingApi.log(run.job), trainingApi.checkpoints(run.job)]);
        if (alive) {
          setLog(lines);
          setCheckpoints(ckpts);
        }
      } catch {
        /* the status line says what is wrong */
      }
    };
    void load();
    const timer = window.setInterval(load, 3000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [run.job]);
  const active = ACTIVE.has(run.state);
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>{run.job}</h2>
        <span>
          {run.state} · {run.hostLabel}
        </span>
      </div>
      <div className="progress">
        <div className="progress-bar" style={{ width: `${progressFraction(run.step, run.totalSteps) * 100}%` }} />
      </div>
      <p className="panel-note">
        step {run.step} / {run.totalSteps}
        {run.loss != null ? ` · loss ${run.loss.toFixed(4)}` : ""} · {run.message}
      </p>
      <div className="control-row">
        {active ? (
          <button className="danger" onClick={() => onAction(() => trainingApi.stop(run.job))}>
            Stop
          </button>
        ) : (
          <>
            <button onClick={() => onAction(() => trainingApi.resume(run.job, null))}>Resume</button>
            <label className="train-inline">
              to step
              <input type="number" value={moreSteps} onChange={(e) => setMoreSteps(Number(e.target.value))} />
            </label>
            <button onClick={() => onAction(() => trainingApi.resume(run.job, moreSteps))}>Train further</button>
            {run.kind === "remote" ? (
              <button onClick={() => onAction(() => trainingApi.fetchRun(run.job))}>
                Fetch checkpoints here{run.fetchedAt ? ` (last ${run.fetchedAt})` : ""}
              </button>
            ) : null}
            <button onClick={() => onAction(() => trainingApi.remove(run.job))}>Forget this run</button>
          </>
        )}
      </div>
      <h3>Checkpoints on this machine</h3>
      {checkpoints.length === 0 ? (
        <p className="panel-note">
          {run.kind === "remote" ? "None here yet: Fetch copies them from the training machine." : "None yet."}
        </p>
      ) : (
        <table className="train-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Contract</th>
              <th>Reload check</th>
              <th>Exports</th>
              <th>Path</th>
            </tr>
          </thead>
          <tbody>
            {checkpoints.map((c) => (
              <tr key={c.path}>
                <td>{c.step}</td>
                <td>{c.hasContract ? "yes" : "MISSING"}</td>
                <td>
                  {c.reloadCheck
                    ? c.reloadCheck.ok
                      ? `identical (max diff ${c.reloadCheck.max_abs_diff ?? 0})`
                      : `FAILED ${c.reloadCheck.error ?? ""}`
                    : "—"}
                </td>
                <td>{c.exports.join(", ")}</td>
                <td className="train-path">{c.path}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <h3>Log</h3>
      <pre className="log-tail train-log">{log.join("\n")}</pre>
    </section>
  );
}

export function TrainingPage() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [exports, setExports] = useState<TrainingExport[]>([]);
  const [form, setForm] = useState<TrainingForm>(DEFAULT_FORM);
  const [check, setCheck] = useState<ExportCheck | null>(null);
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [error, setError] = useState("");
  const [selectedRun, setSelectedRun] = useState("");
  const [newHost, setNewHost] = useState({ label: "", sshTarget: "", repoDir: "", python: ".venv/bin/python" });
  const [extraCamera, setExtraCamera] = useState("");

  const refresh = useCallback(async () => {
    try {
      setStatus(await trainingApi.status());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
    trainingApi.exports().then(setExports, (e) => setError(String(e)));
    const timer = window.setInterval(refresh, 3000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (form.exports.length === 0) {
      setCheck(null);
      return;
    }
    trainingApi.check(form.exports, form.hand).then(setCheck, (e) => setError(String(e)));
  }, [form.exports, form.hand]);

  const act = (fn: () => Promise<unknown>) => {
    setError("");
    fn().then(refresh, (e) => setError(e instanceof Error ? e.message : String(e)));
  };

  const fps = exports.find((e) => form.exports.includes(e.name))?.fps ?? 0;
  const cameraSizes = check?.commonCameras ?? {};
  const problems = useMemo(
    () => [...formProblems(form, cameraSizes), ...(check && !check.ok ? check.problems : [])],
    [form, cameraSizes, check]
  );
  const firstExport = form.exports[0] ?? "";
  const runs = status?.runs ?? [];
  const run = runs.find((r) => r.job === selectedRun) ?? runs[0];
  const set = <K extends keyof TrainingForm>(key: K, value: TrainingForm[K]) => setForm((f) => ({ ...f, [key]: value }));
  const hosts = status?.hosts ?? [];
  const hostGpus = status?.gpus?.[form.hostId]?.gpus;

  // The default machine (a100-19) may not be listed by this gateway: fall back to this machine.
  useEffect(() => {
    if (hosts.length && !hosts.some((h) => h.id === form.hostId)) setForm((f) => ({ ...f, hostId: "local", gpus: [] }));
  }, [hosts, form.hostId]);

  // Keep the picked GPUs to ones the chosen machine has (first free one when none is left).
  useEffect(() => {
    if (!hostGpus) return;
    setForm((f) => {
      const gpus = gpusFor(f.gpus, hostGpus);
      return gpus.join(",") === f.gpus.join(",") ? f : { ...f, gpus };
    });
  }, [hostGpus]);

  // Suggest held-out evaluation episodes when the first export is picked.
  const chosenEpisodes = exports.filter((e) => form.exports.includes(e.name)).reduce((n, e) => Math.max(n, e.episodes), 0);
  useEffect(() => {
    setForm((f) => (f.holdout || !chosenEpisodes ? f : { ...f, holdout: suggestHoldout(chosenEpisodes) }));
  }, [chosenEpisodes]);

  return (
    <div className="page-stack">
      <PageHeader
        title="Training"
        subtitle="train a policy on QC-gated exports, in poses relative to the TCP; every checkpoint says what it was trained on"
      />
      {error ? <div className="command-error">{error}</div> : null}
      <section className="panel">
        <div className="panel-heading">
          <h2>1 · Data</h2>
          <span>exports in outputs/exports</span>
        </div>
        <p className="panel-note">
          Only exports written with the QC gate (11-16 and later) can train: an older export says nothing about which
          episodes passed QC. Poses stay in the camera world in the export and become relative to the TCP when training
          reads them (<a href={userGuideUrl("infer-relative")} target="_blank" rel="noreferrer">guide §8</a>).
        </p>
        <table className="train-table">
          <thead>
            <tr>
              <th />
              <th>Export</th>
              <th>Episodes</th>
              <th>Frames</th>
              <th>Hands</th>
              <th>QC gate</th>
              <th>Usable</th>
            </tr>
          </thead>
          <tbody>
            {exports.map((e) => (
              <tr key={e.name} className={e.usable ? "" : "train-unusable"}>
                <td>
                  <input
                    type="checkbox"
                    disabled={!e.usable}
                    checked={form.exports.includes(e.name)}
                    onChange={(ev) =>
                      set(
                        "exports",
                        ev.target.checked ? [...form.exports, e.name] : form.exports.filter((n) => n !== e.name)
                      )
                    }
                  />
                </td>
                <td>{e.name}</td>
                <td>{e.episodes}</td>
                <td>{e.frames}</td>
                <td>{e.hands.join(", ")}</td>
                <td>{e.qcGated ? "yes" : "no"}</td>
                <td title={e.problems.join("\n")}>{e.usable ? "yes" : e.problems[0]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="control-row">
          <label className="train-inline">
            hand
            <select value={form.hand} onChange={(e) => set("hand", e.target.value as "right" | "left")}>
              <option value="right">right</option>
              <option value="left">left</option>
            </select>
          </label>
          <label className="train-inline" title="Not trained on; offline evaluation on the Rollout page uses them">
            held-out episodes
            <input value={form.holdout} placeholder="none" onChange={(e) => set("holdout", e.target.value)} />
          </label>
        </div>
        <p className="panel-note">
          Held-out episodes are not trained on (nor in the normaliser&apos;s statistics). Offline evaluation on them says
          how the policy does on motions it has not seen; on trained episodes it only says how well it memorised.
        </p>
        {check?.problems.map((p) => (
          <p key={p} className="command-error">
            {p}
          </p>
        ))}
        {check?.warnings.map((w) => (
          <p key={w} className="panel-note level-warn">
            {w}
          </p>
        ))}
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>2 · Cameras</h2>
          <span>cam_10 + the fixed cameras you pick, each with its crop</span>
        </div>
        <p className="panel-note">
          The crop removes what carries no task information (room, people's heads, image edges); the same crop is
          applied to the live frame at inference. A fixed camera is the one input that differs between a person
          recording and the arm executing, and the one that ties the policy to this station.
        </p>
        {form.cameras.map((cam, i) => (
          <CameraEditor
            key={cam.name}
            exportName={firstExport}
            hand={form.hand}
            name={cam.name}
            crop={cam.crop}
            size={cameraSizes[cam.name]}
            longSide={form.imageLongSide}
            removable={cam.name !== "cam_10"}
            onCrop={(crop) => set("cameras", form.cameras.map((c, j) => (j === i ? { ...c, crop } : c)))}
            onRemove={() => set("cameras", form.cameras.filter((_, j) => j !== i))}
          />
        ))}
        <div className="control-row">
          <select value={extraCamera} onChange={(e) => setExtraCamera(e.target.value)}>
            <option value="">add a fixed camera…</option>
            {Object.keys(cameraSizes)
              .filter((name) => !form.cameras.some((c) => c.name === name))
              .map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
          </select>
          <button
            disabled={!extraCamera}
            onClick={() => {
              set("cameras", [...form.cameras, { name: extraCamera, crop: null }]);
              setExtraCamera("");
            }}
          >
            Add
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>3 · Policy</h2>
          <span>{form.policy === "pi05" ? "pi0.5 (LoRA on lerobot/pi05_base)" : "ACT"}</span>
        </div>
        <div className="control-row">
          <label className="train-inline">
            policy
            <select
              value={form.policy}
              onChange={(e) => setForm((f) => ({ ...f, ...policyDefaults(e.target.value as Policy) }))}
            >
              <option value="act">ACT (small, one GPU)</option>
              <option value="pi05">pi0.5 (3B vision-language-action, LoRA, several GPUs)</option>
            </select>
          </label>
        </div>
        {form.policy === "pi05" ? (
          <>
            <p className="panel-note">
              pi0.5 is prompted with the task in words; the checkpoint keeps the text and Rollout sends the same. It
              predicts the same relative-TCP chunk as ACT. Its chunk is sampled from noise; inference always gives it the
              same noise, so offline, online and replay compare exactly. The base weights and the PaliGemma tokenizer
              must be in the training machine&apos;s Hugging Face cache (training runs offline).
            </p>
            <div className="train-grid">
              <label className="train-field">
                <span>Task (prompt)</span>
                <input value={form.task} placeholder="pick up the part and place it" onChange={(e) => set("task", e.target.value)} />
              </label>
              <NumberField label="LoRA rank (0 = all weights)" value={form.loraR} onChange={(v) => set("loraR", v)} />
              <label className="train-field">
                <span>Base checkpoint</span>
                <input value={form.pretrained} placeholder="lerobot/pi05_base" onChange={(e) => set("pretrained", e.target.value)} />
              </label>
            </div>
          </>
        ) : null}
        <p className="panel-note">
          Action: the next {form.horizon} TCP targets relative to the TCP now, every {form.frameStride} frames
          {fps ? ` (${stepHz(fps, form.frameStride).toFixed(0)} Hz, ${chunkSeconds(fps, form.frameStride, form.horizon).toFixed(2)} s)` : ""},
          plus the gripper width. State: the last {form.history} TCP poses relative to now, the width
          {form.wrench ? ", the BOX external wrench" : ""}. These are starting values, not tuned for a task.
        </p>
        <div className="train-grid">
          <NumberField label="Steps" value={form.steps} onChange={(v) => set("steps", v)} />
          <NumberField
            label={form.gpus.length > 1 ? `Batch per GPU (×${form.gpus.length} = ${form.batchSize * form.gpus.length})` : "Batch size"}
            value={form.batchSize}
            onChange={(v) => set("batchSize", v)}
          />
          <NumberField label="Chunk length" value={form.horizon} onChange={(v) => set("horizon", v)} />
          <NumberField label="Executed per call" value={form.nActionSteps} onChange={(v) => set("nActionSteps", v)} />
          <NumberField label="Frame stride" value={form.frameStride} onChange={(v) => set("frameStride", v)} />
          <NumberField label="Pose history" value={form.history} onChange={(v) => set("history", v)} />
          <NumberField label="Image long side" value={form.imageLongSide} onChange={(v) => set("imageLongSide", v)} />
          <NumberField label="Checkpoint every" value={form.saveFreq} onChange={(v) => set("saveFreq", v)} />
          <NumberField label="Data workers" value={form.numWorkers} onChange={(v) => set("numWorkers", v)} />
          <label className="train-field">
            <span>Run name</span>
            <input value={form.jobName} placeholder={`${form.policy}_${form.hand}_<time>`} onChange={(e) => set("jobName", e.target.value)} />
          </label>
        </div>
        <div className="control-row">
          <label className="train-inline">
            <input type="checkbox" checked={form.wrench} onChange={(e) => set("wrench", e.target.checked)} />
            external wrench as input (the arm BOX's payload model must then be in force for Rollout)
          </label>
          <label className="train-inline">
            <input type="checkbox" checked={form.imageAug} onChange={(e) => set("imageAug", e.target.checked)} />
            image augmentation
          </label>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>4 · Machines and GPUs</h2>
          <span>
            {form.gpus.length ? `GPU ${form.gpus.join(", ")} on ` : ""}
            {hosts.find((h) => h.id === form.hostId)?.label ?? form.hostId}
          </span>
        </div>
        <TrainingMachines
          hosts={hosts}
          gpus={status?.gpus ?? {}}
          hostId={form.hostId}
          selected={form.gpus}
          onHost={(id) => {
            setProbe(null);
            setForm((f) => ({ ...f, hostId: id, gpus: id === "local" ? [] : [0] }));
          }}
          onGpus={(gpus) => set("gpus", gpus)}
        />
        <p className="panel-note">
          One GPU runs the trainer directly; several run one process per GPU (accelerate) with the batch per GPU above.
          {status?.recorderConnected ? " The recorder is connected: on a Thor this machine's GPU serves the cameras." : ""}
        </p>
        <div className="control-row">
          <button onClick={() => act(async () => setProbe(await trainingApi.probe(form.hostId)))}>Check machine</button>
          {hosts.find((h) => h.id === form.hostId)?.source === "page" ? (
            <button onClick={() => act(() => trainingApi.removeHost(form.hostId).then(() => set("hostId", "local")))}>
              Remove
            </button>
          ) : null}
        </div>
        {probe ? (
          <p className={probe.ready ? "panel-note" : "command-error"}>
            {probe.ok
              ? `${probe.ready ? "Ready" : "Not ready"}: python ${probe.python}, torch ${probe.torch ?? "missing"}, ` +
                `GPU ${probe.cuda ? `${probe.gpu} (${probe.gpu_mem_gb} GB)` : "none"}, lerobot ${probe.lerobot ? "ok" : "missing"}, ` +
                `video ${probe.torchcodec ? "torchcodec" : probe.av ? "pyav" : "none"}, ffmpeg ${probe.ffmpeg ? "ok" : "missing"}`
              : probe.error}
          </p>
        ) : null}
        <details>
          <summary>Add a remote machine (ssh)</summary>
          <p className="panel-note">
            The directory is used only for training: the code this gateway runs is copied into it before each run and
            files that are not part of it are deleted, so it must be empty (or one this page set up). Its python needs
            torch with CUDA and this repo's dependencies.
          </p>
          <div className="train-grid">
            {(["label", "sshTarget", "repoDir", "python"] as const).map((k) => (
              <label className="train-field" key={k}>
                <span>{{ label: "Name", sshTarget: "user@host", repoDir: "Directory", python: "Python (in the directory)" }[k]}</span>
                <input value={newHost[k]} onChange={(e) => setNewHost({ ...newHost, [k]: e.target.value })} />
              </label>
            ))}
          </div>
          <button onClick={() => act(() => trainingApi.addHost(newHost))}>Add machine</button>
        </details>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>5 · Start</h2>
          <span>{problems.length ? `${problems.length} to fix` : "ready"}</span>
        </div>
        {problems.map((p) => (
          <p key={p} className="panel-note level-warn">
            {p}
          </p>
        ))}
        <button
          disabled={problems.length > 0}
          onClick={() =>
            act(async () => {
              const result = await trainingApi.start(startRequest(form));
              setSelectedRun(result.run.job);
            })
          }
        >
          Start training
        </button>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>Runs</h2>
          <span>{runs.length}</span>
        </div>
        <table className="train-table">
          <thead>
            <tr>
              <th>Run</th>
              <th>State</th>
              <th>Step</th>
              <th>Loss</th>
              <th>Machine</th>
              <th>Started</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.job} className={r.job === run?.job ? "train-selected" : ""} onClick={() => setSelectedRun(r.job)}>
                <td>{r.job}</td>
                <td>{r.state}</td>
                <td>
                  {r.step} / {r.totalSteps}
                </td>
                <td>{r.loss != null ? r.loss.toFixed(4) : ""}</td>
                <td>{r.hostLabel}</td>
                <td>{r.startedAt}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      {run ? <RunDetail key={run.job} run={run} onAction={act} /> : null}
    </div>
  );
}
