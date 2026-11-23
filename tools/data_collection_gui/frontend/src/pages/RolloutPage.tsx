import { useCallback, useEffect, useMemo, useState } from "react";
import { PageHeader, userGuideUrl } from "../shared/ui";
import {
  type OfflineResult,
  type PreviewLatest,
  type ResultRow,
  type RolloutCheckpoint,
  type RolloutStatus,
  rolloutApi
} from "./rolloutApi";
import { ArmGripperPanel } from "./ArmGripperPanel";
import { cameraBlocker, executeBlockers, recorderConnecting, recorderNotStreaming } from "./rolloutExecute";
import { boundsOf, pathOf, stepSeries } from "./rolloutPlot";
import { type TrainingExport, trainingApi } from "./trainingApi";

const W = 360;
const H = 220;

function ErrorCurves({ result }: { result: OfflineResult }) {
  const model = stepSeries(result.error.pos_mm.by_step);
  const still = stepSeries(result.not_moving_baseline_pos_mm.by_step);
  const box = boundsOf([...model, ...still, [1, 0]], 0.05);
  if (!box) return null;
  return (
    <figure className="rollout-figure">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="position error by chunk step">
        <rect x={0} y={0} width={W} height={H} className="rollout-plot-bg" />
        <path d={pathOf(still, box, W, H)} className="rollout-line-still" />
        <path d={pathOf(model, box, W, H)} className="rollout-line-pred" />
      </svg>
      <figcaption>
        Position error (mm) by chunk step: <span className="rollout-key-pred">policy</span> against{" "}
        <span className="rollout-key-still">"not moving"</span>; y from 0 to {box.y1.toFixed(0)} mm.
      </figcaption>
    </figure>
  );
}

function FrameView({ result }: { result: OfflineResult }) {
  const [i, setI] = useState(0);
  const f = result.series[Math.min(i, result.series.length - 1)];
  if (!f) return null;
  // Top view (x, y) of the recording's camera world: where the TCP is, what the policy predicts, what the hand did.
  const pts = [f.tcp, ...f.pred, ...f.truth];
  const box = boundsOf(pts, 0.15);
  return (
    <figure className="rollout-figure">
      {box ? (
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="predicted and recorded TCP path">
          <rect x={0} y={0} width={W} height={H} className="rollout-plot-bg" />
          <path d={pathOf([f.tcp, ...f.truth], box, W, H)} className="rollout-line-truth" />
          <path d={pathOf([f.tcp, ...f.pred], box, W, H)} className="rollout-line-pred" />
          <circle
            cx={((f.tcp[0] - box.x0) / (box.x1 - box.x0)) * W}
            cy={H - ((f.tcp[1] - box.y0) / (box.y1 - box.y0)) * H}
            r={4}
            className="rollout-dot"
          />
        </svg>
      ) : null}
      <input type="range" min={0} max={result.series.length - 1} value={i} onChange={(e) => setI(Number(e.target.value))} />
      <figcaption>
        Episode {f.episode}, frame {f.frame}: <span className="rollout-key-pred">predicted</span> and{" "}
        <span className="rollout-key-truth">recorded</span> TCP over the chunk, seen from above (x, y of the camera world);
        error at the chunk's end {f.pos_err_mm_end ?? "—"} mm; width predicted {(f.pred_width.at(-1) ?? 0).toFixed(3)} m,
        recorded {(f.truth_width.at(-1) ?? 0).toFixed(3)} m.
      </figcaption>
    </figure>
  );
}

function OfflineDetail({ path }: { path: string }) {
  const [result, setResult] = useState<OfflineResult | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    rolloutApi.result(path).then(setResult, (e) => setError(String(e)));
  }, [path]);
  if (error) return <p className="command-error">{error}</p>;
  if (!result) return <p className="panel-note">Loading…</p>;
  const v = result.view_vs_online_path_mm;
  return (
    <div>
      {result.trained_on_this_export ? (
        <p className="panel-note level-warn">
          This checkpoint was trained on this export: the numbers say how well it fits its training data, not how it
          does on new demonstrations.
        </p>
      ) : null}
      <div className="summary-grid">
        <div className="metric">
          <span>Position error</span>
          <strong>
            {result.error.pos_mm.mean.toFixed(1)} mm (p95 {result.error.pos_mm.p95.toFixed(1)})
          </strong>
        </div>
        <div className="metric">
          <span>"Not moving" baseline</span>
          <strong>{result.not_moving_baseline_pos_mm.mean.toFixed(1)} mm</strong>
        </div>
        <div className="metric">
          <span>Rotation error</span>
          <strong>{result.error.rot_deg.mean.toFixed(2)}°</strong>
        </div>
        <div className="metric">
          <span>Width error</span>
          <strong>{result.error.width_mm.mean.toFixed(2)} mm</strong>
        </div>
        <div className="metric">
          <span>Training images vs online images</span>
          <strong>
            max {v.max.toFixed(1)} mm (mean {v.mean.toFixed(2)})
          </strong>
        </div>
        <div className="metric">
          <span>Frames</span>
          <strong>
            {result.frames} (every {result.stride})
          </strong>
        </div>
      </div>
      <p className="panel-note">{v.note}</p>
      <div className="rollout-figures">
        <ErrorCurves result={result} />
        <FrameView result={result} />
      </div>
    </div>
  );
}

function PreviewLive({ record, running }: { record: string; running: boolean }) {
  const [latest, setLatest] = useState<PreviewLatest>({});
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    const load = () =>
      rolloutApi.latest(record).then(
        (l) => alive && (setLatest(l), setTick((t) => t + 1)),
        () => undefined
      );
    void load();
    if (!running) return () => void (alive = false);
    const timer = window.setInterval(load, 1000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [record, running]);
  const targets = latest.targets_tcp_mm ?? [];
  const plan = latest.plan;
  const joints = latest.joints;
  if (latest.blocked) {
    return (
      <div className="rollout-live">
        <p className="command-error">Preview blocked, no prediction: {latest.blocked}</p>
      </div>
    );
  }
  return (
    <div className="rollout-live">
      {latest.step != null ? (
        <img className="train-preview" alt="what the policy sees" src={rolloutApi.previewImageUrl(record, tick)} />
      ) : (
        <p className="panel-note">No prediction yet.</p>
      )}
      {latest.step != null ? (
        <>
          <p className="panel-note">
            step {latest.step} · {latest.latency_ms} ms per prediction · width {latest.width?.toFixed(3)} m
            {latest.wrench ? ` · wrench ${latest.wrench.map((v) => v.toFixed(2)).join(", ")}` : ""}
          </p>
          <table className="train-table">
            <thead>
              <tr>
                <th>Chunk step</th>
                <th>Target in the TCP frame (mm)</th>
                <th>Width (m)</th>
              </tr>
            </thead>
            <tbody>
              {[0, Math.floor(targets.length / 2), targets.length - 1]
                .filter((k, j, all) => k >= 0 && all.indexOf(k) === j)
                .map((k) => (
                  <tr key={k}>
                    <td>{k + 1}</td>
                    <td>{(targets[k] ?? []).map((v) => v.toFixed(1)).join(", ")}</td>
                    <td>{latest.widths?.[k]?.toFixed(3)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
          {plan ? (
            <p className={plan.ok ? "panel-note" : "command-error"}>
              Execution plan check: {plan.ok ? "would pass" : "would stop"} — IK {plan.ik_ok.filter(Boolean).length}/
              {plan.ik_ok.length}, environment gap {plan.min_env_gap_mm ?? "—"} mm
              {plan.reasons.length ? `; ${plan.reasons.join("; ")}` : ""}
              {joints?.source === "read" ? ` (joints read ${joints.age_s} s before this prediction)` : ""}
            </p>
          ) : joints?.source === "read" ? (
            <p className="command-error">
              No plan check: the arm's joints were last read {joints.age_s} s ago
              {joints.error ? ` (${joints.error})` : ""}; a check needs joints at most 5 s old.
            </p>
          ) : (
            <p className="panel-note">No plan check: the arm's joints were not read for this preview.</p>
          )}
          {plan && joints?.error ? <p className="panel-note">Last joint read failed: {joints.error}</p> : null}
        </>
      ) : null}
    </div>
  );
}

export function RolloutPage() {
  const [status, setStatus] = useState<RolloutStatus | null>(null);
  const [checkpoints, setCheckpoints] = useState<RolloutCheckpoint[]>([]);
  const [exports, setExports] = useState<TrainingExport[]>([]);
  const [results, setResults] = useState<ResultRow[]>([]);
  const [checkpoint, setCheckpoint] = useState("");
  const [exportName, setExportName] = useState("");
  const [episodes, setEpisodes] = useState("");
  const [stride, setStride] = useState(15);
  const [arm, setArm] = useState("right");
  const [rate, setRate] = useState(2);
  const [readJoints, setReadJoints] = useState(false);
  const [stage, setStage] = useState(1);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [moveConfirmOpen, setMoveConfirmOpen] = useState(false);
  const [selected, setSelected] = useState<ResultRow | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [error, setError] = useState("");
  // the set of not-streaming sensors whose dialog was closed: a new set (a BOX that drops out later) opens it again
  const [downDismissed, setDownDismissed] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [s, r] = await Promise.all([rolloutApi.status(), rolloutApi.results()]);
      setStatus(s);
      setResults(r);
      const job = s.jobs[0];
      if (job) setLog(await rolloutApi.log(job.id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
    rolloutApi.checkpoints().then((c) => {
      setCheckpoints(c);
      setCheckpoint((cur) => cur || c[0]?.path || "");
    }, (e) => setError(String(e)));
    trainingApi.exports().then((e) => {
      const usable = e.filter((x) => x.usable);
      setExports(usable);
      setExportName((cur) => cur || usable[0]?.name || "");
    }, () => undefined);
    const timer = window.setInterval(refresh, 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const act = (fn: () => Promise<unknown>) => {
    setError("");
    fn().then(refresh, (e) => setError(e instanceof Error ? e.message : String(e)));
  };
  const ckpt = checkpoints.find((c) => c.path === checkpoint);
  const running = status?.jobs.find((j) => j.state === "running");
  const activePreview = status?.jobs.find((j) => j.kind === "preview");
  const runPath = (p: string | undefined) => (p ?? "").split("/runs/")[1] ?? p ?? "";
  const previewReplayed = Boolean(
    ckpt && results.some((r) => r.kind === "preview" && r.replayCheck?.ok && runPath(r.checkpoint) === runPath(ckpt.path))
  );
  const blockers = executeBlockers({ status, checkpoint: ckpt, previewReplayed });
  const openCameras = status?.recorderCameras ?? [];
  const forInference = Boolean(status?.recorderConnected && openCameras.length);
  const wrongCameras = cameraBlocker(status, ckpt);
  const connecting = recorderConnecting(status);
  const down = recorderNotStreaming(status);
  const downNames = [...(down.noBox ? ["no BOX"] : down.boxes), ...down.cameras];
  const downKey = downNames.join(", ");
  useEffect(() => {
    if (!downKey) setDownDismissed("");
  }, [downKey]);
  // the jobs that move the arm: Stop brakes it to rest
  const executing = running?.kind === "execute" || running?.kind === "tostart" ? running : undefined;
  const episodeList = useMemo(
    () => episodes.split(/[\s,]+/).filter(Boolean).map(Number).filter((n) => Number.isInteger(n) && n >= 0),
    [episodes]
  );

  return (
    <div className="page-stack">
      <PageHeader
        title="Rollout"
        subtitle="run a trained policy on recordings, on the live cameras and BOX with the arm still, and — by stages, with a person at the e-stop — on the arm"
      />
      {error ? <div className="command-error">{error}</div> : null}
      <section className="panel">
        <div className="panel-heading">
          <h2>What this page can do now</h2>
          <span>
            <a href={userGuideUrl("infer-inputs")} target="_blank" rel="noreferrer">
              input spec (guide §8)
            </a>
          </span>
        </div>
        <ol className="rollout-stages">
          {(status?.stages ?? []).map((s) => (
            <li key={s.id} className={s.status === "available" ? "rollout-stage-on" : "rollout-stage-off"}>
              <strong>{s.label}</strong> — {s.status}
            </li>
          ))}
        </ol>
        <p className="panel-note">
          Executing a policy on the arm (section 3) starts only by a person with a hand on the e-stop, who confirms
          the dialog, after a preview of the same checkpoint passed its Replay check.
        </p>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>Checkpoint</h2>
          <span>{checkpoints.length} on this machine</span>
        </div>
        <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
          {checkpoints.map((c) => (
            <option key={c.path} value={c.path}>
              {c.run} · step {c.step} · {c.hand} · {c.cameras.join("+")}
              {c.wrench ? " · wrench" : ""}
              {c.policyServer ? " · on policy server" : ""}
            </option>
          ))}
        </select>
        {ckpt ? (
          <p className="panel-note">
            Trained on {ckpt.exports.join(", ") || "?"}; reload check{" "}
            {ckpt.reloadCheck ? (ckpt.reloadCheck.ok ? "identical" : "FAILED") : "not run (an intermediate checkpoint)"}
            {ckpt.hasWorkspace ? "" : "; carries no training workspace, so the plan check cannot bound the arm"}.
            {ckpt.policyServer
              ? ` Its ${ckpt.policyType || "policy"} runs on the policy server ${ckpt.policyServer} (this machine keeps no weights): each prediction takes a round trip, up to 0.30 s while executing, and the server refuses while another process uses its GPU.`
              : " Remote runs appear here after Fetch on the Training page."}
          </p>
        ) : (
          <p className="panel-note">No checkpoint: train one on the Training page (and Fetch it if it trained elsewhere).</p>
        )}
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>1 · Offline prediction</h2>
          <span>through the online code path</span>
        </div>
        <p className="panel-note">
          Each sampled frame of the export goes through what inference does live (full-resolution frame → crop → policy)
          and is compared with what the hand did next, with a policy that predicts "not moving", and with the same
          prediction from the training images.
        </p>
        <div className="control-row">
          <select value={exportName} onChange={(e) => setExportName(e.target.value)}>
            {exports.map((e) => (
              <option key={e.name} value={e.name}>
                {e.name} ({e.episodes} episodes)
              </option>
            ))}
          </select>
          <label className="train-inline">
            episodes
            <input value={episodes} placeholder="all" onChange={(e) => setEpisodes(e.target.value)} />
          </label>
          <label className="train-inline">
            every
            <input type="number" value={stride} onChange={(e) => setStride(Number(e.target.value))} />
            frames
          </label>
          <button
            disabled={!checkpoint || !exportName || Boolean(running)}
            onClick={() => act(() => rolloutApi.offline(checkpoint, exportName, episodeList, stride))}
          >
            Predict offline
          </button>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>2 · Online preview</h2>
          <span>
            {!status?.recorderConnected
              ? "recorder not connected"
              : connecting.pending
                ? connecting.label
                : forInference
                ? `connected for inference: ${openCameras.join(", ")} at 30 Hz`
                : "recorder connected (full session, preview bus ~5 Hz)"}
            {downKey ? ` · not streaming: ${downKey}` : ""}
          </span>
        </div>
        <div className="control-row">
          {!status?.recorderConnected ? (
            <button
              disabled={!ckpt || !ckpt.cameras.length || status?.profile !== "thor" || Boolean(status?.sandbox)}
              onClick={() => ckpt && act(() => rolloutApi.connectForInference(ckpt.cameras))}
            >
              Connect for inference ({ckpt?.cameras.join(", ") || "pick a checkpoint"})
            </button>
          ) : connecting.stuck ? (
            <button onClick={() => act(() => rolloutApi.disconnect())}>
              Cancel connect (not ready after {Math.round(status?.recorder?.elapsedS ?? 0)} s
              {connecting.waiting.length ? `: ${connecting.waiting.join(", ")}` : ""})
            </button>
          ) : connecting.pending ? (
            <button disabled title="Disconnect is available once every camera and BOX streams (or after 60 s)">
              {connecting.label}
            </button>
          ) : forInference ? (
            <button disabled={Boolean(running)} onClick={() => act(() => rolloutApi.disconnect())}>
              Disconnect
            </button>
          ) : null}
          <span className="panel-note">
            Connect for inference opens only the checkpoint's cameras (same 60 fps mode, trigger and exposure as
            recording; the BOXes as usual) and publishes every 2nd frame, so a frame is about 55–90 ms old instead
            of 60–260 ms on a full session's preview. No episode can be recorded until you Disconnect and Connect
            normally (Live Record).
          </span>
        </div>
        {wrongCameras ? <p className="command-error">{wrongCameras}: Disconnect and connect for this checkpoint.</p> : null}
        {downKey ? (
          <p className="command-error">
            Connected, but not streaming: {downKey}. The policy reads the arm gripper BOX's width
            {ckpt?.wrench ? " and wrench" : ""}: power the BOX, then Disconnect and connect again.
          </p>
        ) : null}
        {downKey && downKey !== downDismissed ? (
          <div className="danger-modal-backdrop" role="presentation">
            <div aria-labelledby="rollout-sensor-down" aria-modal="true" className="danger-modal" role="dialog">
              <h3 id="rollout-sensor-down">{down.noBox || down.boxes.length ? "BOX 没有连接" : "相机没有出流"}</h3>
              <p>
                The recorder is connected, but these do not stream: <strong>{downKey}</strong>.
              </p>
              {down.noBox || down.boxes.length ? (
                <p>
                  The policy reads the arm gripper BOX's width{ckpt?.wrench ? " and external wrench" : ""}; without
                  it a preview or execution has no gripper input. Check that the BOX is powered and on the network
                  (a single BOX powered after Connect can still come up on its own: this closes when it streams), or Disconnect and connect again.
                </p>
              ) : (
                <p>Disconnect and connect again; if the camera stays down, see the Device Manager.</p>
              )}
              <div className="danger-modal-actions">
                <button autoFocus onClick={() => setDownDismissed(downKey)} type="button">
                  知道了
                </button>
              </div>
            </div>
          </div>
        ) : null}
        <p className="panel-note">
          The policy reads the recorder's live frames (it never opens a camera) and the arm gripper BOX's width
          {ckpt?.wrench ? " and external wrench" : ""}, with the arm standing still; nothing is sent to the arm. Every
          step's inputs and outputs are recorded, and Replay check feeds them back through the offline path: the
          predictions must be identical. Reading the joints connects to the arm about once a second to read its
          position (connect, read, disconnect: no motion) so each prediction goes through the execution plan check
          at the pose the arm stands in now, also after you move it to another start pose.
        </p>
        <div className="control-row">
          <select value={arm} onChange={(e) => setArm(e.target.value)}>
            <option value="right">right arm</option>
            <option value="left">left arm</option>
          </select>
          <label className="train-inline">
            rate
            <input type="number" step={0.5} value={rate} onChange={(e) => setRate(Number(e.target.value))} />
            Hz
          </label>
          <label className="train-inline">
            <input type="checkbox" checked={readJoints} onChange={(e) => setReadJoints(e.target.checked)} />
            read the arm's joints (for the plan check)
          </label>
          {running?.kind === "preview" ? (
            <button className="danger" onClick={() => act(() => rolloutApi.stop(running.id))}>
              Stop preview
            </button>
          ) : (
            <button
              disabled={!checkpoint || Boolean(running) || !status?.recorderConnected || connecting.pending || Boolean(wrongCameras)}
              onClick={() => act(() => rolloutApi.preview(checkpoint, arm, rate, readJoints))}
            >
              Start preview
            </button>
          )}
        </div>
        <ArmGripperPanel embedded arm={arm} />
        {activePreview ? (
          <PreviewLive record={activePreview.out} running={activePreview.state === "running"} />
        ) : null}
      </section>

      <section className="panel rollout-execute">
        <div className="panel-heading">
          <h2>3 · Execute on the arm</h2>
          <span>{arm} arm</span>
        </div>
        <p className="panel-note">
          <strong>The arm moves.</strong> Start from the start pose with the gripper open and the cubes as recorded,
          stand at the e-stop, and go stage by stage. Every step is checked before it is sent (IK inside the joint
          walls, near where the demonstrations went, 10 mm from carts/table/cameras, per 33 ms step no more than the
          checkpoint's training did — ≤ 15 mm within 100 mm of carts/table/cameras); the arm
          holds where it is on any stop: a step that fails, inputs too old, contact above 15 N, the TCP 20 mm off its
          command, or Stop. Nothing is slowed down: the policy reads speed from its own history.
        </p>
        <div className="control-row">
          <select value={stage} onChange={(e) => setStage(Number(e.target.value))} disabled={Boolean(executing)}>
            {Object.entries(status?.executeStages ?? {}).map(([n, label]) => (
              <option key={n} value={n}>
                stage {n} · {label}
              </option>
            ))}
          </select>
          <span className="panel-note">arm: {arm} (chosen in section 2)</span>
          {executing ? (
            <button className="danger rollout-stop" onClick={() => act(() => rolloutApi.stop(executing.id))}>
              STOP (hold the arm)
            </button>
          ) : (
            <>
              <button
                className="danger"
                disabled={!checkpoint || Boolean(running)}
                title="slowly to a random real start of this checkpoint's training episodes (no policy, gripper untouched)"
                onClick={() => setMoveConfirmOpen(true)}
              >
                Move to a training start
              </button>
              <button
                className="danger"
                disabled={blockers.length > 0}
                title={blockers.join("; ")}
                onClick={() => setConfirmOpen(true)}
              >
                Execute stage {stage}
              </button>
            </>
          )}
        </div>
        <p className="panel-note">
          Move to a training start: one of the poses of this checkpoint's training episodes in their first second,
          drawn at random, with its joints chosen to stay clear of singularities (where the FR3 stops estimating
          force: the yellow light) along the whole episode, with room from the joint walls; the posture may differ
          from the one the arm is in (once: the elbow can swing to the other side), the path in joint space kept 10 mm from
          carts/table/cameras — refused, nothing moving, when no start drawn has such a path. Slow (≤ 0.3 rad/s);
          the path being checked, it runs without the force estimate (a live one still stops on force) and stops
          when the arm is 8 mm off its command (a contact); Stop brakes the arm to rest. Open the gripper yourself
          (the training starts were open).
        </p>
        {executing ? (
          <p className="command-error">Running: {executing.message || "starting"}</p>
        ) : blockers.length ? (
          <ul className="panel-note">
            {blockers.map((b) => (
              <li key={b}>{b}</li>
            ))}
          </ul>
        ) : null}
        {moveConfirmOpen ? (
          <div className="danger-modal-backdrop" role="presentation">
            <div aria-labelledby="rollout-move-confirm" aria-modal="true" className="danger-modal" role="dialog">
              <h3 id="rollout-move-confirm">机器人会动，手放急停</h3>
              <p>
                The {arm} arm moves slowly to a random training start of <strong>{runPath(checkpoint)}</strong>. Keep
                a hand on the e-stop until it stops.
              </p>
              <p>
                It may change the arm's posture (the elbow to the other side, large joint motions, up to ~10 s): clear
                the space around the whole arm and watch the cables, not only the gripper.
              </p>
              <div className="danger-modal-actions">
                <button autoFocus onClick={() => setMoveConfirmOpen(false)} type="button">
                  取消
                </button>
                <button
                  className="danger"
                  onClick={() => {
                    setMoveConfirmOpen(false);
                    act(() => rolloutApi.toStart(checkpoint, arm, true));
                  }}
                  type="button"
                >
                  手已放在急停上，移动
                </button>
              </div>
            </div>
          </div>
        ) : null}
        {confirmOpen ? (
          <div className="danger-modal-backdrop" role="presentation">
            <div aria-labelledby="rollout-execute-confirm" aria-modal="true" className="danger-modal" role="dialog">
              <h3 id="rollout-execute-confirm">机器人会动，手放急停</h3>
              <p>
                The {arm} arm runs stage {stage} of <strong>{runPath(checkpoint)}</strong>. Keep a hand on the
                e-stop until it holds.
              </p>
              <div className="danger-modal-actions">
                <button autoFocus onClick={() => setConfirmOpen(false)} type="button">
                  取消
                </button>
                <button
                  className="danger"
                  onClick={() => {
                    setConfirmOpen(false);
                    act(() => rolloutApi.execute(checkpoint, arm, stage, true));
                  }}
                  type="button"
                >
                  手已放在急停上，执行
                </button>
              </div>
            </div>
          </div>
        ) : null}
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>Jobs</h2>
          <span>{running ? `${running.kind} running` : "idle"}</span>
        </div>
        <table className="train-table">
          <thead>
            <tr>
              <th>Job</th>
              <th>State</th>
              <th>Message</th>
              <th>Started</th>
            </tr>
          </thead>
          <tbody>
            {(status?.jobs ?? []).slice(0, 8).map((j) => (
              <tr key={j.id}>
                <td>{j.id}</td>
                <td>{j.state}</td>
                <td>{j.message}</td>
                <td>{j.startedAt}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {running?.kind === "offline" || running?.kind === "replay" ? (
          <button onClick={() => act(() => rolloutApi.stop(running.id))}>Stop</button>
        ) : null}
        <pre className="log-tail train-log">{log.slice(-30).join("\n")}</pre>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>Results</h2>
          <span>{results.length}</span>
        </div>
        <table className="train-table">
          <thead>
            <tr>
              <th>Kind</th>
              <th>Created</th>
              <th>Checkpoint</th>
              <th>Data</th>
              <th>Outcome</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.path} className={selected?.path === r.path ? "train-selected" : ""}>
                <td>{r.kind}</td>
                <td>{r.created}</td>
                <td className="train-path">{r.checkpoint?.split("/runs/")[1] ?? r.checkpoint}</td>
                <td>
                  {r.kind === "offline"
                    ? `${r.export} (${r.frames} frames)`
                    : r.kind === "execute"
                      ? `${r.arm} arm, stage ${r.stage ?? "?"}`
                      : `${r.arm} arm, ${r.steps ?? "…"} steps`}
                </td>
                <td>
                  {r.kind === "offline"
                    ? `${r.posMm?.toFixed(1)} mm vs still ${r.stillMm?.toFixed(1)} mm; images Δ ≤ ${r.viewVsOnlineMm?.toFixed(1)} mm`
                    : r.kind === "execute"
                      ? r.result
                        ? `${r.result.completed ? "completed" : `stopped: ${r.result.stopped}`} (${r.result.chunks} chunks, ${r.result.duration_s.toFixed(1)} s)`
                        : "running or interrupted"
                      : r.replayCheck
                      ? `replay ${r.replayCheck.ok ? "identical" : "DIFFERENT"} (max ${r.replayCheck.max_abs_diff.toExponential(1)})`
                      : r.planProblem || "not replayed yet"}
                </td>
                <td>
                  {r.kind === "offline" ? (
                    <button onClick={() => setSelected(r)}>Open</button>
                  ) : (
                    <button disabled={Boolean(running)} onClick={() => act(() => rolloutApi.replay(r.path))}>
                      Replay check
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {selected?.kind === "offline" ? <OfflineDetail key={selected.path} path={selected.path} /> : null}
      </section>
    </div>
  );
}
