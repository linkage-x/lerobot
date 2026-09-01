import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../apiClient";
import { MujocoReplayViewer } from "../MujocoReplayViewer";
import type { MujocoPreview, RolloutLiveFrame } from "../types";

/**
 * The arm as it moves, drawn from the rollout that is moving it.
 *
 * The replay path renders MuJoCo to a file and plays it afterwards, which is right for
 * validating a recorded episode and wrong for watching one happen: by the time there is a file,
 * the moment worth seeing is over. Here the runtime publishes joint angles every step, the
 * gateway keeps the last few hundred, and this component draws them as they arrive.
 *
 * **Why it plays a buffer instead of drawing the newest frame.** Polling every 150 ms while the
 * arm produces 30 frames a second means each response carries four or five frames. Drawing only
 * the last one throws four away and turns smooth motion into a 6 Hz stutter. So frames go into a
 * queue and a timer walks through them at the rate they were produced. The cost is that the
 * picture trails the arm by about one poll -- a fifth of a second, on a motion that takes
 * seconds. The alternative, drawing every frame the instant it arrives, would show them in
 * bursts and be *less* faithful to how the arm actually moved.
 *
 * If the queue grows past a second of motion -- a slow tab, a page that was in the background --
 * playback skips forward rather than falling further behind. A live view that is thirty seconds
 * stale is not a live view, and quietly becoming one is worse than visibly jumping.
 */

const POLL_MS = 150;
// Roughly what the gateway holds. Past this the queue is trimmed from the front: the frames
// worth drawing are the recent ones, and everything is in the trace CSV either way.
const MAX_BUFFERED_FRAMES = 600;

function isBehind(queued: number, fps: number): boolean {
  return queued > Math.max(fps, 1);
}

export function RolloutLiveViewer({
  live,
  fps = 30,
  rolloutIndex
}: {
  live: boolean;
  fps?: number;
  rolloutIndex: number;
}) {
  const [frames, setFrames] = useState<RolloutLiveFrame[]>([]);
  const [cursor, setCursor] = useState(0);
  const [stale, setStale] = useState(false);
  const seqRef = useRef(0);
  const framesRef = useRef<RolloutLiveFrame[]>([]);
  framesRef.current = frames;

  // A new rollout is not a continuation of the last one. Cleared here rather than on the next
  // frame so the canvas does not hold the previous rollout's final pose while the new one homes.
  useEffect(() => {
    setFrames([]);
    setCursor(0);
  }, [rolloutIndex]);

  useEffect(() => {
    if (!live) return undefined;
    let cancelled = false;
    const tick = async () => {
      const payload = await api.fetchRolloutLiveFrames(seqRef.current);
      if (cancelled || !payload) return;
      seqRef.current = payload.seq;
      setStale(!payload.running);
      if (!payload.frames.length) return;
      setFrames((previous) => {
        // `dropped` means the gateway's buffer rolled past us, so what just arrived does not
        // continue what we hold. Start again from the new frames rather than splicing a jump
        // into the middle of the playback.
        const merged = payload.dropped ? payload.frames : [...previous, ...payload.frames];
        return merged.length > MAX_BUFFERED_FRAMES
          ? merged.slice(merged.length - MAX_BUFFERED_FRAMES)
          : merged;
      });
      if (payload.dropped) setCursor(0);
    };
    void tick();
    const timer = window.setInterval(() => void tick(), POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [live]);

  // The player. One tick per frame period, one frame per tick, except when it is behind.
  useEffect(() => {
    if (!live) return undefined;
    const period = 1000 / Math.max(fps, 1);
    const timer = window.setInterval(() => {
      setCursor((current) => {
        const total = framesRef.current.length;
        if (total === 0) return 0;
        const remaining = total - 1 - current;
        if (remaining <= 0) return total - 1;
        // Catching up: jump to a quarter-second behind the newest frame rather than crawling
        // forward one frame at a time from wherever the tab was suspended.
        if (isBehind(remaining, fps)) return Math.max(0, total - 1 - Math.round(fps / 4));
        return current + 1;
      });
    }, period);
    return () => window.clearInterval(timer);
  }, [live, fps]);

  const preview = useMemo<MujocoPreview | null>(() => {
    if (!frames.length) return null;
    return {
      schema_version: 1,
      dataset_root: "",
      cube_mode: "left",
      episode_index: rolloutIndex,
      fps,
      robot_spacing_m: 0,
      frames,
      streaming: true,
      stream_frame_count: frames.length,
      frame_source: "rollout joint state",
      robots: {},
      model: { renderer: "three-webgl", kinematics_path: "/fr3_mujoco_replay/kinematics.json" }
    };
  }, [frames, fps, rolloutIndex]);

  const shown = frames.length ? frames[Math.min(cursor, frames.length - 1)] : null;
  const source = shown?.source ?? "";
  const queued = frames.length ? frames.length - 1 - Math.min(cursor, frames.length - 1) : 0;

  return (
    <div className="rollout-live-viewer">
      <div className="row-actions rollout-live-viewer-head">
        {/* The one thing this view exists to make obvious: who is driving. During a takeover the
            arm is doing what the operator asked, and a rollout that reads as the policy's when
            half of it was the operator's is a rollout that would poison a success rate. */}
        <span className={source === "expert" ? "pill pill-warn" : "pill"}>
          {source === "expert" ? "Operator driving" : source === "policy" ? "Policy driving" : "—"}
        </span>
        <span className="hint">
          {frames.length === 0
            ? stale
              ? "No rollout is publishing frames."
              : "Waiting for the first frame."
            : `step ${shown?.frame_index ?? 0} · ${queued} frame(s) buffered${
                shown?.status && shown.status !== "pass" ? ` · ${shown.status}` : ""
              }`}
        </span>
      </div>
      <MujocoReplayViewer preview={preview} currentFrame={cursor} />
    </div>
  );
}
