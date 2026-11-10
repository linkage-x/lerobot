// The one gateway job behind the load / 6D panels (check-poses, collect, hand-held capture):
// state, progress, the tail of its log, and Stop.
import { useEffect, useRef } from "react";
import { isJobActive, jobPercent, type PayloadJob } from "./payloadCalibration";

const STATE_TEXT: Record<string, string> = {
  running: "进行中",
  fitting: "拟合中",
  done: "完成",
  error: "失败",
  stopped: "已停止",
};

export function PayloadJobView({
  job,
  onStop,
  stopping,
  tail = 14,
}: {
  job: PayloadJob;
  onStop?: () => void;
  stopping?: boolean;
  tail?: number;
}) {
  const logRef = useRef<HTMLPreElement | null>(null);
  const lines = (job.log ?? []).slice(-tail);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [lines.length]);
  const active = isJobActive(job);
  const pct = jobPercent(job);
  const tone = job.state === "done" ? "running" : job.state === "error" ? "error" : active ? "warning" : "idle";
  return (
    <div className={`cali-result-box payload-job payload-job-${job.state ?? "idle"}`}>
      <div className="cali-result-box-head">
        <span className={`cali-badge cali-badge-${tone}`}>{STATE_TEXT[job.state ?? ""] ?? "—"}</span>
        <strong>{job.label ?? job.kind}</strong>
        {job.run || job.name ? <code className="cali-muted">{job.run ?? job.name}</code> : null}
        {active && onStop ? (
          <button className="payload-stop" disabled={stopping} onClick={onStop}>
            {stopping ? "正在停止…" : "停止"}
          </button>
        ) : null}
      </div>
      {pct != null && (
        <div className="cali-solve">
          <div className="progress" aria-label="进度">
            <div className="progress-bar" style={{ width: `${pct}%` }} />
          </div>
          <span className="cali-muted">
            {job.progress?.done}/{job.progress?.total} 次测量 · {pct}%
          </span>
        </div>
      )}
      {job.error ? <div className="cali-result-reason">{job.error}</div> : null}
      {job.kind !== "six_d_zero" &&
        (job.warnings ?? []).map((w) => (
          <p key={w} className="cali-warn">
            {w}
          </p>
        ))}
      {lines.length > 0 && (
        <pre className="payload-log" ref={logRef}>
          {lines.join("\n")}
        </pre>
      )}
    </div>
  );
}
