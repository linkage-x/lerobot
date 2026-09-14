import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { UnattendedKind, UnattendedListEntry, UnattendedPlan, UnattendedRun } from "../types";
import { PageHeader, StatusDot } from "../shared/ui";
import {
  VERDICT_COLORS,
  mapPoints,
  outcomeLabel,
  projectRows,
  runHealth,
  verdictCounts
} from "./unattendedRuns";

/** One page for every run that happens with nobody in the room.
 *
 *  Not a page per experiment. E6-lite and E6-data are the same shape -- expand a seeded schedule,
 *  walk it emitting one row per unit, name the condition you stopped on, park holding the peg --
 *  and the card behind this work set a hard standard for the UI that replaces babysitting a
 *  terminal: **it should have fewer controls than today, not more.** Two pages would be more.
 *
 *  Once the loop runs itself a person has exactly three jobs, and the page is laid out as those
 *  three and nothing else:
 *
 *  **Authorise.** Not a start button: a schedule, expanded and fence-checked before anybody is
 *  asked to approve it, printed the way the runtime's own `--plan-only` prints it. The thing read
 *  is the thing that runs. Signing it makes it read-only.
 *
 *  **Witness.** Two heartbeats and a disk line, because one light cannot tell an arm that stopped
 *  from a recorder that stopped and those need opposite responses. A map with three layers, the
 *  third being *where it halted* -- a night of green dots with no mark at the end says nothing
 *  about why there are only forty. And one row per unit carrying the numbers its verdict came
 *  from, because the acceptance criterion is agreement with a human and a verdict nobody can
 *  check is one they can only accept.
 *
 *  **Brake.** Two of them, presented as two actions. Boundary finishes the unit in flight and
 *  parks holding the peg, so the next run starts from a state the loop's invariant covers.
 *  Immediate stops it where it stands, which is right when something is wrong and wrong otherwise.
 *
 *  And the property that is not a feature but a constraint: **this page is not part of the control
 *  loop.** Everything here reads a directory on the rig. Closing the browser, reloading, losing
 *  the network, restarting the gateway -- none of them touch a run, and reopening the page shows
 *  exactly what it showed before, because there was never any state here to lose.
 */

const POLL_MS = 2000;

type Draft = { kind: string; request: Record<string, string> };

const DEFAULT_REQUESTS: Record<string, Record<string, string>> = {
  terminal_trials: {
    holePose: "0.3599,-0.1333,0.0523",
    offsetsMm: "0,2,3,4,5,6,8,10",
    repeats: "6",
    controlEvery: "4",
    seed: "0",
    searchRingM: "0",
    maxSeconds: "7200"
  },
  auto_collect: {
    cycles: "50",
    seed: "0",
    recoveryFraction: "0",
    placeZ: "0.0550",
    carryZ: "0.1500",
    maxSeconds: "7200"
  }
};

const FIELD_HELP: Record<string, string> = {
  holePose: "Starting estimate only -- reference trials re-read it from contact. Trial 000 is the check: a wrong estimate costs two minutes, not a night.",
  searchRingM: "0 measures the bare capture radius, which is the number the 7 mm ring was sized against. Turning the ring on measures the ring instead.",
  controlEvery: "A reference trial every N offset trials. Each one is a hole reading; you want at least 8 over the run.",
  recoveryFraction: "Fraction of cycles that start from a displaced pose. 0 is the acceptance run.",
  maxSeconds: "0 runs the whole schedule. A budget is cheaper than discovering the rate at 3 a.m."
};

export function UnattendedPage({ api }: { api: DataCollectionGuiApi }) {
  const [kinds, setKinds] = useState<UnattendedKind[]>([]);
  const [runs, setRuns] = useState<UnattendedListEntry[]>([]);
  const [active, setActive] = useState<UnattendedListEntry | null>(null);
  const [selectedId, setSelectedId] = useState<string>("");
  const [run, setRun] = useState<UnattendedRun | null>(null);
  const [draft, setDraft] = useState<Draft>({ kind: "terminal_trials", request: DEFAULT_REQUESTS.terminal_trials });
  const [plan, setPlan] = useState<UnattendedPlan | null>(null);
  const [error, setError] = useState<string>("");
  const [busy, setBusy] = useState(false);
  // Kept in a ref so the poll below never closes over a stale id.
  const selectedRef = useRef(selectedId);
  selectedRef.current = selectedId;

  const refresh = useCallback(async () => {
    const listing = await api.fetchUnattendedRuns();
    if (!listing?.ok) return;
    setKinds(listing.kinds);
    setRuns(listing.runs);
    setActive(listing.active);
    const id = selectedRef.current || listing.active?.id || listing.runs[0]?.id || "";
    if (id !== selectedRef.current) setSelectedId(id);
    if (!id) {
      setRun(null);
      return;
    }
    const detail = await api.fetchUnattendedRun(id);
    if (detail?.ok) setRun(detail.run);
  }, [api]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const guard = async (label: string, call: () => Promise<{ ok: boolean; error?: string }>) => {
    setBusy(true);
    setError("");
    try {
      const result = await call();
      if (!result.ok) setError(`${label}: ${result.error ?? "refused"}`);
      return result;
    } finally {
      setBusy(false);
      await refresh();
    }
  };

  const onPlan = async () => {
    setPlan(null);
    const result = await guard("plan", () => api.planUnattendedRun(draft.kind, draft.request));
    if (result.ok && "plan" in result) setPlan((result as { plan: UnattendedPlan }).plan);
  };

  const onStart = async () => {
    const result = await guard("start", () => api.startUnattendedRun(draft.kind, draft.request));
    if (result.ok && "run" in result) {
      const started = (result as { run: UnattendedRun }).run;
      setSelectedId(started.id);
      setPlan(null);
    }
  };

  const units = useMemo(() => projectRows(run?.rows ?? []), [run]);
  const health = useMemo(() => (run ? runHealth(run) : null), [run]);
  const haltedAt = useMemo(() => {
    if (!run || run.state === "running" || run.state === "starting") return null;
    return units.length ? units[units.length - 1].index : null;
  }, [run, units]);
  const schedule = (run?.plan as UnattendedPlan | undefined)?.schedule ?? [];
  const points = useMemo(() => mapPoints(schedule, units, haltedAt), [schedule, units, haltedAt]);
  const counts = useMemo(() => verdictCounts(units), [units]);
  const running = run?.state === "running" || run?.state === "starting";

  return (
    <div className="page-stack">
      <PageHeader
        title="Unattended Runs"
        subtitle="authorise a schedule, witness it, brake it -- this page is not in the control loop, and closing it changes nothing"
      />
      {error ? <section className="panel"><p style={{ color: "#e53e3e", margin: 0 }}>{error}</p></section> : null}

      {/* ---------------------------------------------------------------- authorise */}
      <section className="panel">
        <div className="panel-heading">
          <h2>Authorise</h2>
          <span>{active ? `${active.id} is ${active.state} -- there is one arm` : "nothing is running"}</span>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 12 }}>
          {kinds.map((kind) => (
            <button
              key={kind.id}
              className={draft.kind === kind.id ? "primary" : ""}
              disabled={busy || Boolean(active)}
              onClick={() => {
                setDraft({ kind: kind.id, request: DEFAULT_REQUESTS[kind.id] ?? {} });
                setPlan(null);
              }}
            >
              {kind.label}
            </button>
          ))}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))", gap: 10 }}>
          {Object.entries(draft.request).map(([key, value]) => (
            <label key={key} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 12, opacity: 0.8 }}>{key}</span>
              <input
                value={value}
                disabled={busy || Boolean(active)}
                onChange={(event) => {
                  setDraft((current) => ({
                    ...current,
                    request: { ...current.request, [key]: event.target.value }
                  }));
                  setPlan(null);
                }}
              />
              {FIELD_HELP[key] ? (
                <span style={{ fontSize: 11, opacity: 0.6 }}>{FIELD_HELP[key]}</span>
              ) : null}
            </label>
          ))}
        </div>
        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <button onClick={() => void onPlan()} disabled={busy || Boolean(active)}>
            Read the plan
          </button>
          {/* Start is only reachable through a plan that was expanded and fence-checked. That is
              what authorisation means here -- a start button that skipped it would be a button
              that moves the arm ten thousand times on an unread schedule. */}
          <button className="primary" onClick={() => void onStart()} disabled={busy || !plan || Boolean(active)}>
            Start this plan
          </button>
        </div>
        {plan ? (
          <div style={{ marginTop: 12 }}>
            <p style={{ margin: "0 0 6px", fontSize: 13 }}>
              <strong>{plan.units}</strong> units · fence {plan.fence.source} · QC passed
            </p>
            <pre
              style={{
                maxHeight: 240,
                overflow: "auto",
                fontSize: 11,
                background: "rgba(0,0,0,0.25)",
                padding: 10,
                margin: 0
              }}
            >
              {plan.text}
            </pre>
          </div>
        ) : null}
      </section>

      {/* ------------------------------------------------------------------ witness */}
      <section className="panel">
        <div className="panel-heading">
          <h2>Witness</h2>
          <span>
            {runs.length ? (
              <select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
                {runs.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.id} — {entry.state} ({entry.unitsDone}/{entry.unitsPlanned || "?"})
                  </option>
                ))}
              </select>
            ) : (
              "no runs yet"
            )}
          </span>
        </div>
        {run ? (
          <>
            <p style={{ margin: "0 0 10px" }}>
              <StatusDot state={run.state} /> <strong>{run.state}</strong> — {outcomeLabel(run)}
            </p>
            {health?.divergence ? (
              // The single most valuable line on the page, and the one a combined status light
              // cannot produce.
              <p style={{ color: "#e53e3e", margin: "0 0 10px", fontWeight: 600 }}>
                ⚠ {health.divergence}
              </p>
            ) : null}
            <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginBottom: 12 }}>
              {(health?.heartbeats ?? []).map((beat) => (
                <span key={beat.label} style={{ fontSize: 13 }}>
                  <StatusDot state={beat.ok ? "pass" : "error"} /> <strong>{beat.label}</strong>{" "}
                  {beat.ageS === null ? "never" : `${Math.round(beat.ageS)}s ago`} — {beat.detail}
                </span>
              ))}
            </div>
            <p style={{ fontSize: 12, opacity: 0.75, margin: "0 0 12px" }}>
              rows land in <code>{run.dir}</code> · log <code>{run.logPath}</code>
            </p>
            <RunMap points={points} />
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", margin: "10px 0" }}>
              {counts.map(([verdict, count]) => (
                <span key={verdict} style={{ fontSize: 13 }}>
                  <span
                    style={{
                      display: "inline-block",
                      width: 10,
                      height: 10,
                      borderRadius: 5,
                      background: VERDICT_COLORS[verdict] ?? "#718096",
                      marginRight: 5
                    }}
                  />
                  {verdict} × {count}
                </span>
              ))}
            </div>
            <div style={{ maxHeight: 280, overflow: "auto" }}>
              <table className="table" style={{ width: "100%", fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>kind</th>
                    <th>verdict</th>
                    <th>readings</th>
                  </tr>
                </thead>
                <tbody>
                  {[...units].reverse().map((unit) => (
                    <tr key={unit.index}>
                      <td>{unit.index}</td>
                      <td>{unit.kind}</td>
                      <td style={{ color: VERDICT_COLORS[unit.verdict] ?? undefined }}>{unit.verdict}</td>
                      <td>{unit.readings.map(([label, value]) => `${label} ${value}`).join(" · ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p style={{ margin: 0, opacity: 0.7 }}>Nothing selected.</p>
        )}
      </section>

      {/* -------------------------------------------------------------------- brake */}
      {run ? (
        <section className="panel">
          <div className="panel-heading">
            <h2>Brake</h2>
            <span>{run.stopRequested ? `stop requested: ${run.stopReason}` : "no stop requested"}</span>
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <button
              disabled={busy || !running || run.stopRequested}
              onClick={() => void guard("stop", () => api.stopUnattendedRun(run.id, "boundary"))}
            >
              Stop at the next boundary
            </button>
            <button
              disabled={busy || !running || !run.stopRequested}
              onClick={() => void guard("release", () => api.releaseUnattendedBrake(run.id))}
            >
              Release
            </button>
            <button
              className="danger"
              disabled={busy || !running}
              onClick={() => {
                // The only confirmation on this page. Everything else here is recoverable; this
                // one leaves the peg wherever it happens to be, which is a state the loop's
                // invariant does not cover and a person has to fix by hand.
                if (window.confirm("Stop where it stands? The peg is left wherever it is.")) {
                  void guard("halt", () => api.stopUnattendedRun(run.id, "now"));
                }
              }}
            >
              Halt now
            </button>
          </div>
          <p style={{ fontSize: 12, opacity: 0.7, marginBottom: 0 }}>
            The boundary stop finishes the {run.kind === "terminal_trials" ? "trial" : "cycle"} in
            flight and parks holding the peg, so the next run starts from a state the loop expects.
            Halting now does not.
          </p>
        </section>
      ) : null}
    </div>
  );
}

function RunMap({ points }: { points: { x: number; y: number; status: string; index: number }[] }) {
  const SIZE = 300;
  const PAD = 18;
  if (!points.length) return <p style={{ opacity: 0.6, fontSize: 13 }}>No points to draw yet.</p>;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  // A degenerate span is a real case -- every terminal trial aims at the same point -- so the
  // scale falls back to a fixed span rather than dividing by zero and drawing nothing.
  const spanX = maxX - minX || 0.05;
  const spanY = maxY - minY || 0.05;
  const project = (point: { x: number; y: number }) => ({
    cx: PAD + ((point.x - minX) / spanX) * (SIZE - 2 * PAD),
    cy: SIZE - PAD - ((point.y - minY) / spanY) * (SIZE - 2 * PAD)
  });
  return (
    <svg width={SIZE} height={SIZE} style={{ background: "rgba(0,0,0,0.2)", borderRadius: 6 }}>
      {points.map((point) => {
        const { cx, cy } = project(point);
        const done = point.status !== "planned";
        return (
          <circle
            key={point.index}
            cx={cx}
            cy={cy}
            r={point.status === "halt" ? 7 : done ? 5 : 3}
            fill={done ? VERDICT_COLORS[point.status] ?? "#718096" : "none"}
            stroke={VERDICT_COLORS[point.status] ?? "#718096"}
            strokeWidth={point.status === "halt" ? 2.5 : 1}
          >
            <title>{`#${point.index} ${point.status}`}</title>
          </circle>
        );
      })}
    </svg>
  );
}
