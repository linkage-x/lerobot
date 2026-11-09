import { useCallback, useEffect, useState } from "react";
import { api } from "../apiClient";
import type { GpuHostEntry } from "../types";
import { StatusDot } from "../shared/ui";
import { buttonsFor, describeService, opNote, type GpuAction } from "./gpuHosts";

const DOT: Record<string, string> = { running: "running", starting: "connecting", stopping: "connecting", stopped: "idle" };

/** What holds each policy host's GPU, and the buttons to start, stop or switch its services.
 *
 * Polled faster while an action runs on a host, so the operator sees the stop, the memory coming
 * free and the start without refreshing. Stop and switch take a second click to confirm: they end
 * a server someone may be using from another machine. */
export function GpuHostPanel({ rolloutLive }: { rolloutLive: boolean }) {
  const [hosts, setHosts] = useState<GpuHostEntry[] | null>(null);
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [sending, setSending] = useState(false);

  const refresh = useCallback(async () => {
    const listing = await api.fetchGpuHosts();
    if (listing) setHosts(listing.hosts);
  }, []);
  const opRunning = hosts?.some((host) => host.status?.op?.state === "running") ?? false;
  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), opRunning ? 1500 : 5000);
    return () => window.clearInterval(timer);
  }, [refresh, opRunning]);

  const run = async (address: string, action: GpuAction, service: string) => {
    const key = `${address}/${action}/${service}`;
    if (action !== "start" && confirm !== key) {
      setConfirm(key);
      return;
    }
    setConfirm("");
    setError("");
    setSending(true);
    const result = await api.gpuHostAction(address, action, service);
    setSending(false);
    if (!result.ok) setError(result.error || `${action} ${service} failed.`);
    await refresh();
  };

  if (hosts === null || hosts.length === 0) return null;
  return (
    <div className="gpu-hosts">
      {hosts.map((host) => {
        if (!host.reachable || !host.status) {
          return (
            <p key={host.address} className="hint">
              GPU services on <code>{host.address}</code>: no agent answering ({host.error}). Start it there with{" "}
              <code>bash run/gpu_services_agent.sh start</code>.
            </p>
          );
        }
        const status = host.status;
        const note = opNote(status.op);
        return (
          <div key={host.address} className="gpu-host">
            <div className="gpu-host-head">
              <strong>
                {status.host} {status.gpu.name}
              </strong>{" "}
              <span className="hint">
                {(status.gpu.used_mb / 1024).toFixed(1)} / {(status.gpu.total_mb / 1024).toFixed(1)} GB in use
              </span>
              <div className="gpu-host-bar" aria-hidden="true">
                <span style={{ width: `${Math.min(100, (100 * status.gpu.used_mb) / status.gpu.total_mb)}%` }} />
              </div>
            </div>
            {status.services.map((svc) => (
              <div key={svc.id} className="policy-server-row">
                <StatusDot state={DOT[svc.state] ?? "idle"} />
                <span title={svc.label}>
                  <code>{svc.id}</code> — {describeService(svc)}
                </span>
                {buttonsFor(status, svc, rolloutLive).map((button) => {
                  const key = `${host.address}/${button.action}/${svc.id}`;
                  return (
                    <button
                      key={button.action}
                      type="button"
                      disabled={sending || !!button.disabledReason}
                      title={button.disabledReason || svc.label}
                      onClick={() => void run(host.address, button.action, svc.id)}
                    >
                      {confirm === key ? `Confirm: ${button.label.toLowerCase()}` : button.label}
                    </button>
                  );
                })}
              </div>
            ))}
            {status.others.map((other) => (
              <p key={other.pid} className="hint">
                Also on the card: pid {other.pid}, {(other.mb / 1024).toFixed(1)} GB — <code>{other.what}</code>
              </p>
            ))}
            {status.training.length > 0 && <p className="hint">Training on this card: {status.training.join(", ")}</p>}
            {note && <p className={status.op?.state === "failed" ? "hint gpu-host-failed" : "hint"}>{note}</p>}
          </div>
        );
      })}
      {error && <div className="banner banner-error">{error}</div>}
    </div>
  );
}
