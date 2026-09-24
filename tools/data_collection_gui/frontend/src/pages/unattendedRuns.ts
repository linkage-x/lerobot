import type { UnattendedRow, UnattendedRun, UnattendedState } from "../types";

/** What the panel decides, kept out of the component so it can be checked without a browser.
 *
 *  The decisions here are the ones the card behind this page asked for, and each of them is a
 *  thing a simpler panel gets wrong:
 *
 *  - A single "it is running" light cannot tell an arm that stopped from a recorder that stopped,
 *    and those need opposite responses. So there are separate heartbeats and, more importantly,
 *    an explicit reading of the moment they *diverge*.
 *  - Rows accumulating in a process are not data. The disk line is its own heartbeat.
 *  - `complete` and `crashed` both mean the process is gone. Collapsing them is how a night that
 *    stopped at 2 a.m. gets read as a night that finished.
 */

/** How long a heartbeat may go quiet before the panel calls it stalled, in seconds.
 *
 *  Generous, because the unrecorded legs of a cycle take tens of seconds and a trial's descent is
 *  slow by design (the terminal servo walks at 0.02 m/s). Anything tighter would cry wolf through
 *  every normal transfer, and a warning that fires during normal operation is a warning nobody
 *  reads at 2 a.m.
 */
export const HEARTBEAT_STALE_S = 180;

export type Heartbeat = {
  label: string;
  /** Seconds since this thing last did something, or null when it never has. */
  ageS: number | null;
  ok: boolean;
  detail: string;
};

export type RunHealth = {
  heartbeats: Heartbeat[];
  /** Set when the heartbeats disagree, which is the reading the operator actually needs. */
  divergence: string;
};

export function isTerminalState(state: UnattendedState): boolean {
  return state === "complete" || state === "halted" || state === "crashed";
}

/** The one-line verdict on a finished run. `crashed` is never folded into anything. */
export function outcomeLabel(run: Pick<UnattendedRun, "state" | "summary">): string {
  switch (run.state) {
    case "complete":
      return "finished the schedule";
    case "halted":
      return `stopped on ${String(run.summary?.haltedOn ?? "an unnamed condition")}`;
    case "crashed":
      return "the process is gone and no summary was written -- nobody has read why yet";
    case "running":
      return "running";
    case "starting":
      return "starting";
    default:
      return "planned";
  }
}

export function runHealth(run: UnattendedRun, nowS: number = Date.now() / 1000): RunHealth {
  const rowAge = run.lastRowAgeS;
  const startedAge = run.startedAt ? Math.max(0, nowS - run.startedAt) : null;
  const running = run.state === "running" || run.state === "starting";

  const controller: Heartbeat = {
    label: "controller",
    ageS: run.alive ? startedAge : null,
    ok: run.alive,
    detail: run.alive
      ? `pid ${run.pid ?? "?"} alive`
      : run.pid
        ? `pid ${run.pid} is gone`
        : "never started"
  };
  const disk: Heartbeat = {
    label: "last write",
    ageS: rowAge,
    ok: rowAge !== null && (!running || rowAge <= HEARTBEAT_STALE_S),
    detail:
      rowAge === null
        ? "nothing has been written yet"
        : `${run.rowsTotal} rows, ${run.unitsDone}/${run.unitsPlanned || "?"} done`
  };

  // The reading that matters. A live process that has stopped writing is the case a single light
  // cannot express, and it is the one that costs a night: the arm is still moving and nothing is
  // being recorded.
  let divergence = "";
  if (running && rowAge !== null && rowAge > HEARTBEAT_STALE_S) {
    divergence = `the process is alive but nothing has been written for ${Math.round(rowAge)}s -- it is moving and not recording`;
  } else if (!run.alive && run.state === "crashed") {
    divergence = "the process is gone and the run never wrote a summary";
  } else if (running && rowAge === null && startedAge !== null && startedAge > HEARTBEAT_STALE_S) {
    divergence = `started ${Math.round(startedAge)}s ago and has produced nothing`;
  }

  return { heartbeats: [controller, disk], divergence };
}

export type UnitRow = {
  index: number;
  kind: string;
  verdict: string;
  /** The raw readings behind the verdict, so a row can be argued with rather than believed. */
  readings: [string, string][];
};

const READING_KEYS: [string, string, (value: number) => string][] = [
  ["aboveTargetMm", "above target", (v) => `${v.toFixed(2)} mm`],
  ["settleMm", "settle", (v) => `${v.toFixed(2)} mm`],
  ["offsetMm", "offset", (v) => `${v.toFixed(1)} mm`],
  ["widthNormalized", "width", (v) => v.toFixed(3)],
  ["referenceStepMm", "hole moved", (v) => `${v.toFixed(2)} mm`],
  ["lateralErrorMm", "lateral", (v) => `${v.toFixed(2)} mm`],
  ["dxMm", "dx", (v) => `${v.toFixed(1)}`],
  ["dyMm", "dy", (v) => `${v.toFixed(1)}`],
  ["dzMm", "dz", (v) => `${v.toFixed(1)} mm`],
  ["widthLifted", "width lifted", (v) => v.toFixed(3)],
  ["contactZ", "contact z", (v) => v.toFixed(4)],
  ["verifyWidth", "verify", (v) => v.toFixed(3)]
];

/** One row per unit of work, carrying the numbers the verdict came from.
 *
 *  The readings travel with the verdict because the card's acceptance criterion is agreement with
 *  a human, and a verdict a person cannot check is one they can only accept.
 */
export function projectRows(rows: UnattendedRow[]): UnitRow[] {
  const out: UnitRow[] = [];
  for (const row of rows) {
    const kind = String(row.kind ?? "");
    if (kind !== "trial" && kind !== "cycle") continue;
    if (row.ok === false) continue;
    const readings: [string, string][] = [];
    for (const [key, label, format] of READING_KEYS) {
      const value = row[key];
      if (typeof value === "number" && Number.isFinite(value)) readings.push([label, format(value)]);
    }
    // A grasp-envelope trial that left the peg unrecoverable is its own outcome on the map: the
    // knock-over boundary is one of the four numbers the sweep exists to find.
    const verdict = String(row.verdict ?? "");
    out.push({
      index: typeof row.index === "number" ? row.index : out.length,
      kind: String(row.trialKind ?? row.cycleKind ?? kind),
      verdict: row.pegDisturbed === true ? `${verdict}+disturbed` : verdict,
      readings
    });
  }
  return out;
}

export function verdictCounts(units: UnitRow[]): [string, number][] {
  const counts = new Map<string, number>();
  for (const unit of units) counts.set(unit.verdict || "?", (counts.get(unit.verdict || "?") ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

export type MapPoint = {
  x: number;
  y: number;
  /** "planned" until the unit runs, then its verdict; "halt" for the one it stopped on. */
  status: string;
  index: number;
};

/** Planned points, completed points coloured by verdict, and where it stopped.
 *
 *  The third layer is the one that is usually missing and the one worth the most in the morning:
 *  a night of green dots with no mark at the end says nothing about why there are only forty.
 */
export function mapPoints(
  schedule: Record<string, unknown>[],
  units: UnitRow[],
  haltedAt: number | null
): MapPoint[] {
  const verdictByIndex = new Map(units.map((unit) => [unit.index, unit.verdict]));
  const points: MapPoint[] = [];
  schedule.forEach((spec, order) => {
    const xyz = (spec.placeXyz ?? spec.aimXyz) as number[] | undefined;
    const index = typeof spec.index === "number" ? spec.index : order;
    if (!Array.isArray(xyz) || xyz.length < 2) return;
    const verdict = verdictByIndex.get(index);
    points.push({
      x: Number(xyz[0]),
      y: Number(xyz[1]),
      status: haltedAt === index ? "halt" : (verdict ?? "planned"),
      index
    });
  });
  return points;
}

export const VERDICT_COLORS: Record<string, string> = {
  planned: "#4a5568",
  seated: "#38a169",
  held: "#38a169",
  standing: "#d69e2e",
  empty: "#e53e3e",
  changed: "#dd6b20",
  slip: "#e53e3e",
  ambiguous: "#805ad5",
  halt: "#e53e3e",
  contact: "#d69e2e",
  "held+disturbed": "#805ad5",
  "empty+disturbed": "#9b2c2c",
  "contact+disturbed": "#9b2c2c"
};

/** The four envelope numbers from a grasp-envelope summary row, as label/value pairs.
 *
 *  An edge the grid ran out before reaching is shown as "≥", and misses inside the capture radius
 *  are named: a radius that tolerated an outlier says which one.
 */
export function envelopeReadout(summary: Record<string, unknown> | null | undefined): [string, string][] {
  if (!summary || !("xyCaptureRadiusMm" in summary)) return [];
  const mm = (value: unknown) => (typeof value === "number" ? `${value.toFixed(1)} mm` : "—");
  const interval = summary.graspDzIntervalMm;
  const open = (Array.isArray(summary.graspDzOpen) ? summary.graspDzOpen : [false, false]) as boolean[];
  const centre = summary.centre as { n?: number; held?: number } | undefined;
  const verify = summary.verify as { n?: number; held?: number } | undefined;
  const pooled = summary.xyCaptureHeld as { n?: number; held?: number } | undefined;
  const misses = (Array.isArray(summary.xyMissesInsideCapture) ? summary.xyMissesInsideCapture : []) as number[][];
  let capture = mm(summary.xyCaptureRadiusMm);
  if (capture !== "—") {
    if (summary.xyCaptureAtGridEdge === true) capture = `≥ ${capture} (grid edge)`;
    if (pooled?.n) capture += ` · ${pooled.held}/${pooled.n} held`;
    if (misses.length) capture += ` · missed at ${misses.map(([dx, dy]) => `(${dx},${dy})`).join(" ")}`;
  }
  const knock =
    summary.knockOverRadiusMm === null && typeof summary.xyDisturbed === "number"
      ? `none in grid (${summary.xyDisturbed} scattered)`
      : mm(summary.knockOverRadiusMm);
  return [
    ["XY capture radius", capture],
    [
      "grasp dz interval",
      Array.isArray(interval) && interval.length === 2
        ? `${open[0] ? "≤ " : ""}${interval[0]} … ${open[1] ? "≥ " : ""}${interval[1]} mm`
        : "—"
    ],
    ["contact at or below dz", mm(summary.contactBelowDzMm)],
    ["knock-over from radius", knock],
    ["centre held", centre?.n ? `${centre.held}/${centre.n}` : "—"],
    ["verify grasps held", verify?.n ? `${verify.held}/${verify.n}` : "—"]
  ];
}
