import { useMemo, useState } from "react";

import type {
  DemoLandingPoint,
  RolloutGeometry,
  RolloutLandmarks,
  RolloutOutcomeEntry,
  TableWindow
} from "../types";
import { buildLandingPoints, formatMm, type PlottedPoint, pointFill, stageFill } from "./landingMapPoints";
import { windowForPlot } from "./tableWindow";

/** A top-down map of where rollouts actually put the gripper, over where the demonstrations did.
 *
 * The reason this is a map and not a success rate: a rate answers "how often", and the question
 * a partially working policy raises is "where". Three rollouts that fail at three corners of the
 * workspace and three that fail at the same spot have the same rate and completely different
 * causes -- the first is coverage, the second is a systematic offset -- and no scalar can tell
 * them apart. The demonstrations are drawn underneath for the same reason: a landing point is
 * only interpretable relative to the region the policy was ever shown.
 *
 * Plotted in the robot's own base frame rather than in camera pixels. The base frame is the one
 * the dataset, the runtime and this page already agree on, so no point on this map depends on a
 * camera calibration that could be stale. The camera can still appear *underneath* them, but
 * only the other way round: the still is re-projected into this plot's base-frame window, and
 * when there is no calibration to do that with, the backdrop is simply absent.
 */

const SIZE = 460;
const PAD_LEFT = 52;
const PAD_BOTTOM = 40;
const PAD_TOP = 16;
const PAD_RIGHT = 16;
const GRID_STEP_M = 0.05;

export function RolloutLandingMap({
  landmarks,
  entries,
  pendingIndex,
  pendingGeometry,
  checkpointId,
  tableViewUrl,
  backgroundLabel = "side camera"
}: {
  landmarks: RolloutLandmarks;
  entries: RolloutOutcomeEntry[];
  pendingIndex: number;
  pendingGeometry?: RolloutGeometry;
  checkpointId: string;
  /** Builds the URL of the camera still re-projected onto exactly this rectangle of table.
   *  Absent until the camera has been aligned, and then the map draws its grid alone. */
  tableViewUrl?: (window: TableWindow, width: number, height: number) => string;
  backgroundLabel?: string;
}) {
  const demoPoints: DemoLandingPoint[] = landmarks.points ?? [];
  const [backgroundLoaded, setBackgroundLoaded] = useState(false);

  const rolloutPoints = useMemo<PlottedPoint[]>(
    () => buildLandingPoints(entries, pendingIndex, pendingGeometry),
    [entries, pendingIndex, pendingGeometry]
  );

  const frame = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    demoPoints.forEach((point) => {
      xs.push(point.graspXyz[0]);
      ys.push(point.graspXyz[1]);
    });
    rolloutPoints.forEach((point) => {
      xs.push(point.x);
      ys.push(point.y);
    });
    if (landmarks.hole) {
      xs.push(landmarks.hole[0]);
      ys.push(landmarks.hole[1]);
    }
    if (xs.length === 0) return null;
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    // One span for both axes. Two would make the plot fill its box more neatly and would also
    // make a 20 mm error look like a 40 mm one along whichever axis happened to be tighter,
    // which is the single thing this plot must never do.
    const span = Math.max(maxX - minX, maxY - minY, 0.05) * 1.18;
    const centreX = (minX + maxX) / 2;
    const centreY = (minY + maxY) / 2;
    const plotWidth = SIZE - PAD_LEFT - PAD_RIGHT;
    const plotHeight = SIZE - PAD_TOP - PAD_BOTTOM;
    const scale = Math.min(plotWidth, plotHeight) / span;
    const originX = PAD_LEFT + plotWidth / 2;
    const originY = PAD_TOP + plotHeight / 2;
    return {
      span,
      centreX,
      centreY,
      // Base x runs rightward and base y upward, matching the axis labels and the scene-reset
      // map. The camera backdrop is re-projected into this same convention, so changing either
      // half without the other silently mirrors the table under the points.
      toScreenX: (x: number) => originX + (x - centreX) * scale,
      toScreenY: (y: number) => originY - (y - centreY) * scale,
      toWorldX: (screenX: number) => centreX + (screenX - originX) / scale,
      toWorldY: (screenY: number) => centreY + (originY - screenY) / scale,
      scale
    };
  }, [demoPoints, rolloutPoints, landmarks.hole]);

  if (!frame) {
    return (
      <p className="hint">
        No landing points yet. The map fills in as rollouts finish — each one adds the point where
        its gripper closed.
      </p>
    );
  }

  // The plot rectangle expressed in base metres, which is what the backdrop has to cover.
  const plotWidth = SIZE - PAD_LEFT - PAD_RIGHT;
  const plotHeight = SIZE - PAD_TOP - PAD_BOTTOM;
  const viewWindow: TableWindow = windowForPlot(frame, {
    left: PAD_LEFT,
    top: PAD_TOP,
    right: SIZE - PAD_RIGHT,
    bottom: SIZE - PAD_BOTTOM
  });
  const backgroundUrl = tableViewUrl ? tableViewUrl(viewWindow, plotWidth, plotHeight) : "";

  const gridLines: { value: number; axis: "x" | "y" }[] = [];
  const halfSpan = frame.span / 2;
  for (const axis of ["x", "y"] as const) {
    const centre = axis === "x" ? frame.centreX : frame.centreY;
    const first = Math.ceil((centre - halfSpan) / GRID_STEP_M) * GRID_STEP_M;
    for (let value = first; value <= centre + halfSpan; value += GRID_STEP_M) {
      gridLines.push({ value, axis });
    }
  }

  const hole = landmarks.hole;
  const radius = landmarks.graspRadiusM;
  const graded = rolloutPoints.filter((point) => point.outcome !== "pending");
  const successes = graded.filter((point) => point.outcome === "success").length;
  // The ramp replaces the success/failure key only once something on the map carries a stage;
  // a log full of records graded the old way would otherwise get a legend for colours it has
  // no points in.
  const staged = graded.filter(
    (point) => point.stage !== undefined && point.terminalStage !== undefined
  );
  const terminalStage = staged.length
    ? Math.max(...staged.map((point) => point.terminalStage as number))
    : 0;

  return (
    <div className="landing-map">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="Rollout landing points">
        {backgroundUrl ? (
          <image
            href={backgroundUrl}
            x={PAD_LEFT}
            y={PAD_TOP}
            width={plotWidth}
            height={plotHeight}
            preserveAspectRatio="none"
            className="table-view-image"
            onError={() => setBackgroundLoaded(false)}
            onLoad={() => setBackgroundLoaded(true)}
            style={{ visibility: backgroundLoaded ? "visible" : "hidden" }}
          >
            <title>{`${backgroundLabel}, projected onto the table plane`}</title>
          </image>
        ) : null}
        <rect
          x={PAD_LEFT}
          y={PAD_TOP}
          width={plotWidth}
          height={plotHeight}
          className="landing-map-frame"
        />
        {gridLines.map(({ value, axis }) => {
          const isX = axis === "x";
          const screen = isX ? frame.toScreenX(value) : frame.toScreenY(value);
          if (isX && (screen < PAD_LEFT || screen > SIZE - PAD_RIGHT)) return null;
          if (!isX && (screen < PAD_TOP || screen > SIZE - PAD_BOTTOM)) return null;
          return (
            <g key={`${axis}-${value.toFixed(3)}`}>
              <line
                x1={isX ? screen : PAD_LEFT}
                x2={isX ? screen : SIZE - PAD_RIGHT}
                y1={isX ? PAD_TOP : screen}
                y2={isX ? SIZE - PAD_BOTTOM : screen}
                className="landing-map-grid"
              />
              <text
                x={isX ? screen : PAD_LEFT - 6}
                y={isX ? SIZE - PAD_BOTTOM + 14 : screen + 3}
                className="landing-map-tick"
                textAnchor={isX ? "middle" : "end"}
              >
                {(value * 1000).toFixed(0)}
              </text>
            </g>
          );
        })}

        {hole && radius && (
          <>
            {/* The annulus the demonstrations actually cover. Anything outside it is a placement
                the policy was never shown, and its failure is a coverage fact, not a policy one. */}
            <circle
              cx={frame.toScreenX(hole[0])}
              cy={frame.toScreenY(hole[1])}
              r={radius.min * frame.scale}
              className="landing-map-band"
            />
            <circle
              cx={frame.toScreenX(hole[0])}
              cy={frame.toScreenY(hole[1])}
              r={radius.max * frame.scale}
              className="landing-map-band"
            />
          </>
        )}

        {demoPoints.map((point) => (
          <circle
            key={`demo-${point.episode}`}
            cx={frame.toScreenX(point.graspXyz[0])}
            cy={frame.toScreenY(point.graspXyz[1])}
            r={3}
            className="landing-map-demo"
          >
            <title>{`Demo episode ${point.episode} — insert descent ${formatMm(point.descentM)}`}</title>
          </circle>
        ))}

        {hole && (
          <g className="landing-map-hole">
            <line
              x1={frame.toScreenX(hole[0]) - 7}
              x2={frame.toScreenX(hole[0]) + 7}
              y1={frame.toScreenY(hole[1])}
              y2={frame.toScreenY(hole[1])}
            />
            <line
              x1={frame.toScreenX(hole[0])}
              x2={frame.toScreenX(hole[0])}
              y1={frame.toScreenY(hole[1]) - 7}
              y2={frame.toScreenY(hole[1]) + 7}
            />
            <title>Hole, measured as the mean of every demonstration&apos;s release point</title>
          </g>
        )}

        {rolloutPoints.map((point) => {
          const cx = frame.toScreenX(point.x);
          const cy = frame.toScreenY(point.y);
          const fill = pointFill(point);
          return (
            <g key={point.key}>
              {/* A ring in the plot's own background colour, so a dot that lands on an earlier
                  one still reads as two dots instead of silently replacing it. */}
              <circle cx={cx} cy={cy} r={point.closed ? 7 : 8} className="landing-map-halo" />
              {/* Filled when the gripper closed, hollow when it only reached: the same dot in
                  two states, because "did not grip" and "gripped and dropped it" are different
                  failures at the same coordinates. */}
              <circle
                cx={cx}
                cy={cy}
                r={point.closed ? 5.5 : 6.5}
                fill={point.closed ? fill : "none"}
                stroke={fill}
                strokeWidth={point.closed ? 1 : 2}
                className={point.outcome === "pending" ? "landing-map-pending" : undefined}
              />
              <title>{point.title}</title>
            </g>
          );
        })}

        <text x={SIZE / 2} y={SIZE - 6} className="landing-map-axis" textAnchor="middle">
          base x (mm)
        </text>
        <text
          x={12}
          y={SIZE / 2}
          className="landing-map-axis"
          textAnchor="middle"
          transform={`rotate(-90 12 ${SIZE / 2})`}
        >
          base y (mm)
        </text>
      </svg>

      <div className="landing-map-legend">
        <span><i className="dot demo" /> demo grasp ({demoPoints.length})</span>
        {staged.length ? (
          <>
            <span className="landing-map-ramp">
              <i className="dot" style={{ background: stageFill(1, terminalStage) }} />
              {Array.from({ length: terminalStage - 1 }, (_, index) => (
                <i
                  key={index + 2}
                  className="dot"
                  style={{ background: stageFill(index + 2, terminalStage) }}
                />
              ))}
              {` stage 1 → ${terminalStage} (success)`}
            </span>
          </>
        ) : (
          <>
            <span><i className="dot success" /> success ({successes})</span>
            <span><i className="dot failure" /> failure</span>
          </>
        )}
        <span><i className="dot aborted" /> aborted</span>
        <span><i className="dot pending" /> not graded</span>
        <span><i className="dot hollow" /> gripper never closed</span>
      </div>
      <p className="hint">
        Landing points for <code>{checkpointId || "this checkpoint"}</code>, in the robot base
        frame. The two dashed circles are the closest and furthest the demonstrations ever grasped
        from the hole
        {radius ? ` (${formatMm(radius.min)}–${formatMm(radius.max)})` : ""}; a failure outside
        them is a placement the policy was never trained on.
        {staged.length
          ? " Dot colour is how far along the task's precondition chain the rollout got, so a" +
            " cluster of one colour is a cluster of one failure mode."
          : ""}
      </p>
    </div>
  );
}
