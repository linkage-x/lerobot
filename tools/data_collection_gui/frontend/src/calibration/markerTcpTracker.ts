// The laser tracker's side of the marker->TCP panel (E1p), kept pure so it can
// be tested without a DOM.
//
// A camera pivot sample and an E1p capture are the same physical act: the TCP
// insert seated in the socket, the gripper turned about it. Recorded with the
// SMR on its plate and the tracker on, one sweep serves both, continuously: the
// camera fit reads every frame, and E1p reads every frame the tracker says was
// seated (the SMR on its sphere about the socket). No pauses. What this module
// decides is whether the samples can be solved yet, and says why not in the
// operator's terms.
import type { MarkerTcpSample, MarkerTcpSession } from "../types";
import { POSES_TO_CERTIFY } from "./trackerMount";

/** Rotation cell E1p pools frames into; certification counts cells, not frames. */
export const E1P_ATTITUDE_CELL_DEG = 5;

const TRACKER_SESSION_ID = /\blt_\d{8}_\d{6}\b/;

export function liveTrackerSessionId(detail: string | undefined): string {
  return TRACKER_SESSION_ID.exec(detail ?? "")?.[0] ?? "";
}

/** Saved samples of this BOX and this clamping that were recorded with the tracker on. */
export function e1pSamples(samples: MarkerTcpSample[], boxId: string, condition: string): MarkerTcpSample[] {
  const wanted = condition.trim();
  return samples.filter(
    (sample) =>
      sample.status === "saved" &&
      Boolean(sample.laserTracker) &&
      (sample.boxId || sample.side) === boxId &&
      sample.condition === wanted,
  );
}

export type E1pChoice = { boxId: string; conditions: { condition: string; samples: number }[] };

/**
 * What E1p can be run on, from the samples themselves.
 *
 * Not from the live BOX list: E1p can only run after Disconnect (the stream
 * lands then), and after Disconnect the live list is back to the static rows
 * that carry no identity -- a dropdown built from it would be empty exactly
 * when it is needed.
 */
export function e1pChoices(samples: MarkerTcpSample[]): E1pChoice[] {
  const byBox = new Map<string, Map<string, number>>();
  for (const sample of samples) {
    if (sample.status !== "saved" || !sample.laserTracker) continue;
    const box = sample.boxId || sample.side;
    if (!box) continue;
    const conditions = byBox.get(box) ?? new Map<string, number>();
    conditions.set(sample.condition, (conditions.get(sample.condition) ?? 0) + 1);
    byBox.set(box, conditions);
  }
  return [...byBox].map(([boxId, conditions]) => ({
    boxId,
    conditions: [...conditions].map(([condition, count]) => ({ condition, samples: count })),
  }));
}

type RecordingLike = {
  state: string;
  laserTracker?: boolean;
  laserTrackerReady?: boolean;
  laserTrackerHomed?: boolean;
  laserTrackerBeamBroken?: boolean;
  laserTrackerDetail?: string;
};

const DISCONNECTED = new Set(["idle", "error"]);

export type TrackerLink = { dot: "running" | "warning" | "idle"; text: string };

/** Whether the samples recorded right now will also be E1p captures. */
export function trackerLink(recording: RecordingLike): TrackerLink {
  if (DISCONNECTED.has(recording.state)) {
    return {
      dot: "idle",
      text: "录制器没连接。要同时录 E1p：到「采集」页打开跟踪仪开关再 Connect。",
    };
  }
  if (!recording.laserTracker) {
    return {
      dot: "idle",
      text:
        "这次 Connect 没开跟踪仪，样本只能做相机 pivot。要顺便测 E1p：Disconnect，打开跟踪仪开关，重新 Connect。",
    };
  }
  if (!recording.laserTrackerHomed) {
    return {
      dot: "warning",
      text: `跟踪仪这次会话还没 Home 成功（${recording.laserTrackerDetail || "等待中"}）。先把 SMR 放进 home 窝等 Home 成功，否则距离不可信，「录制样本」会被拒绝。`,
    };
  }
  if (recording.laserTrackerBeamBroken) {
    return {
      dot: "warning",
      text: "跟踪仪 Home 之后断过光，当前距离不是绝对的：请把 SMR 放回 home 窝重新 Home（放回后自动 Home），否则「录制样本」会被拒绝。",
    };
  }
  if (!recording.laserTrackerReady) {
    return {
      dot: "warning",
      text: `跟踪仪还没锁定 SMR（${recording.laserTrackerDetail || "等待中"}）。锁定之前「录制样本」会被拒绝。`,
    };
  }
  return {
    dot: "running",
    text: "跟踪仪已锁定：这次录的样本同时是 E1p 采集（SMR 留在钢片上）。",
  };
}

export type E1pReadiness = { canRun: boolean; reason: string; samples: MarkerTcpSample[] };

/** Whether the E1p button can do anything, and what to do first if not. */
export function e1pReadiness(args: {
  session: MarkerTcpSession;
  boxId: string;
  condition: string;
  stationPath: string;
  recording: RecordingLike;
}): E1pReadiness {
  const { session, boxId, condition, stationPath, recording } = args;
  const samples = e1pSamples(session.samples, boxId, condition);
  const refuse = (reason: string): E1pReadiness => ({ canRun: false, reason, samples });
  if (session.pendingSampleId) return refuse("还有样本正在录制，先保存或丢弃。");
  if (session.stage === "solving") return refuse("解算进行中，等它结束。");
  if (!boxId) return refuse("先选 BOX。");
  if (!condition.trim()) return refuse("先填条件：一个条件 = 一次夹持 = 一个球面。");
  if (!samples.length) {
    return refuse(`${boxId} · ${condition.trim()} 还没有带跟踪仪的 saved 样本。`);
  }
  // The stream seals and lands at Disconnect. Samples from this Connect are on
  // disk and correct, and not one of them can be solved until then -- the
  // state an operator most needs told, because nothing else looks wrong.
  const live = liveTrackerSessionId(recording.laserTrackerDetail);
  if (
    !DISCONNECTED.has(recording.state) &&
    recording.laserTracker &&
    samples.some((sample) => !sample.trackerSessionId || sample.trackerSessionId === live)
  ) {
    return refuse("这批样本的跟踪仪 session 要 Disconnect 才落地：录完先到「采集」页 Disconnect，再回来解。");
  }
  if (!stationPath) {
    return refuse("还没有 station：用同一跟踪仪站位的驻点集先解出来（跟踪仪不动的话，先录 pivot 后解站位也行）。");
  }
  return {
    canRun: true,
    reason:
      `将用 ${samples.length} 段样本的连续扫动跑 E1p：跟踪仪判定插件在窝内的帧都参与比较；` +
      `认证要 ≥ ${POSES_TO_CERTIFY} 个 ${E1P_ATTITUDE_CELL_DEG}° 姿态格，全程断光的样本会被列出、不进比较。`,
    samples,
  };
}
