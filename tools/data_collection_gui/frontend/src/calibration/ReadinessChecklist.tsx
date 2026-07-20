// Collection readiness checklist (spec §2). Renders the assembled readiness
// items; items whose datum the backend does not expose are shown as
// "unavailable" (never silently "complete") and carry a TODO note.

import type { DotState, ReadinessItem, ReadinessState } from "./types";

const STATE_DOT: Record<ReadinessState, DotState> = {
  complete: "running",
  pending: "idle",
  warning: "warning",
  failed: "error",
  unavailable: "idle",
};

const STATE_ICON: Record<ReadinessState, string> = {
  complete: "✓",
  pending: "…",
  warning: "!",
  failed: "✕",
  unavailable: "—",
};

const STATE_TEXT: Record<ReadinessState, string> = {
  complete: "完成",
  pending: "待完成",
  warning: "注意",
  failed: "未通过",
  unavailable: "不可用",
};

export function ReadinessChecklist({ items }: { items: ReadinessItem[] }) {
  const complete = items.filter((i) => i.state === "complete").length;
  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>采集准备度</h2>
        <span>
          {complete} / {items.length} 已完成
        </span>
      </div>
      <div className="cali-readiness">
        {items.map((item) => (
          <div className={`cali-readiness-row cali-readiness-${item.state}`} key={item.id}>
            <span className="cali-readiness-icon" aria-hidden="true">
              <span className={`status-dot status-${STATE_DOT[item.state]}`} />
              {STATE_ICON[item.state]}
            </span>
            <span className="cali-readiness-label">{item.label}</span>
            <span className="cali-readiness-detail">{item.detail}</span>
            <span className={`cali-readiness-state cali-readiness-state-${item.state}`}>
              {STATE_TEXT[item.state]}
            </span>
            {item.action && <div className="cali-readiness-action">{item.action}</div>}
            {item.todo && <small className="cali-readiness-todo">TODO：{item.todo}</small>}
          </div>
        ))}
      </div>
    </section>
  );
}
