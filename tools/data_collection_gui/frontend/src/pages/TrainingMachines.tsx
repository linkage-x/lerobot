// The Training page's machines: every training host with its GPUs (memory, load, whose processes), and
// the GPUs the next run takes on the chosen one. Readings come with the page's status poll; the gateway
// refreshes them in the background (training_runs.TrainingManager.gpu_status).
import type { Gpu, GpuHostStatus, TrainingHost } from "./trainingApi";
import { gpuBusy, gpuMemText } from "./trainingGpus";

type Props = {
  hosts: TrainingHost[];
  gpus: Record<string, GpuHostStatus>;
  hostId: string;
  selected: number[];
  onHost: (id: string) => void;
  onGpus: (gpus: number[]) => void;
};

function GpuRow({ gpu, picked, pickable, onToggle }: { gpu: Gpu; picked: boolean; pickable: boolean; onToggle: () => void }) {
  const used = gpu.memUsedMiB ?? 0;
  const total = gpu.memTotalMiB ?? 0;
  const busy = gpuBusy(gpu);
  return (
    <tr className={busy ? "gpu-busy" : ""}>
      <td>
        {pickable ? <input type="checkbox" checked={picked} onChange={onToggle} aria-label={`GPU ${gpu.index}`} /> : null}
      </td>
      <td>{gpu.index}</td>
      <td>{gpu.name.replace("NVIDIA ", "")}</td>
      <td>
        <div className="gpu-bar" title={gpuMemText(gpu)}>
          <div className="gpu-bar-fill" style={{ width: `${total ? Math.min(100, (100 * used) / total) : 0}%` }} />
        </div>
        <span className="gpu-mem">{gpuMemText(gpu)}</span>
      </td>
      <td>{gpu.utilPct == null ? "—" : `${gpu.utilPct.toFixed(0)}%`}</td>
      <td className="gpu-procs">
        {gpu.processes.length === 0
          ? "free"
          : gpu.processes.map((p) => (
              <span key={p.pid} title={p.command}>
                {p.job ? `run ${p.job}` : `${p.user} pid ${p.pid}`}
                {p.memMiB != null ? ` (${(p.memMiB / 1024).toFixed(1)} GB)` : ""}
              </span>
            ))}
      </td>
    </tr>
  );
}

export function TrainingMachines({ hosts, gpus, hostId, selected, onHost, onGpus }: Props) {
  return (
    <div className="train-machines">
      {hosts.map((host) => {
        const reading = gpus[host.id] ?? { state: "probing" };
        const chosen = host.id === hostId;
        const list = reading.gpus ?? [];
        const toggle = (index: number) =>
          onGpus(selected.includes(index) ? selected.filter((g) => g !== index) : [...selected, index].sort((a, b) => a - b));
        return (
          <div key={host.id} className={`train-machine${chosen ? " chosen" : ""}`}>
            <div className="train-machine-head">
              <label className="train-inline">
                <input type="radio" name="training-host" checked={chosen} onChange={() => onHost(host.id)} />
                <strong>{host.label}</strong>
              </label>
              <span className="panel-note">
                {host.kind === "remote" ? `${host.sshTarget}${host.port !== 22 ? `:${host.port}` : ""} · ${host.repoDir}` : "the gateway's machine"}
              </span>
              <span className={reading.state === "ok" ? "panel-note" : "level-warn"}>
                {reading.state === "ok"
                  ? `${list.length} GPU${list.length === 1 ? "" : "s"}, ${list.filter((g) => !gpuBusy(g)).length} free`
                  : reading.state === "probing"
                    ? "reading GPUs…"
                    : `unreachable: ${reading.error ?? ""}`}
              </span>
              {chosen && list.length > 1 ? (
                <span className="train-machine-pick">
                  <button onClick={() => onGpus(list.filter((g) => !gpuBusy(g)).map((g) => g.index))}>all free</button>
                  <button onClick={() => onGpus([])}>none</button>
                </span>
              ) : null}
            </div>
            {host.note ? <p className="panel-note">{host.note}</p> : null}
            {list.length ? (
              <table className="data-table gpu-table">
                <thead>
                  <tr>
                    <th />
                    <th>GPU</th>
                    <th>Model</th>
                    <th>Memory</th>
                    <th>Load</th>
                    <th>Processes</th>
                  </tr>
                </thead>
                <tbody>
                  {list.map((gpu) => (
                    <GpuRow
                      key={gpu.index}
                      gpu={gpu}
                      pickable={chosen}
                      picked={chosen && selected.includes(gpu.index)}
                      onToggle={() => toggle(gpu.index)}
                    />
                  ))}
                </tbody>
              </table>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
