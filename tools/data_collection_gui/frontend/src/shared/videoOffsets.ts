/**
 * Converting between a replay timeline's time axis and a `<video>` element's `currentTime`.
 *
 * The two are not the same clock. Timeline timestamps come from the episode's parquet rows and
 * restart at 0 for every episode; `video.currentTime` is local to whatever file the gateway
 * served. `cameraVideoOffsetsS[key]` is the bridge: the timeline time at which that file's own
 * zero sits. It is positive for a GMSL2 recording, whose per-episode camera file starts a little
 * *after* the shared t0, and negative for a LeRobot v3 dataset, whose file holds the whole chunk
 * and starts `from_timestamp` seconds *before* this episode's first frame.
 *
 * A missing or non-finite entry means zero, which is the right answer for a file that holds
 * exactly this episode and nothing else -- and is exactly what makes a wrong one so quiet: a
 * chunk of twenty episodes all seek to the same clip and only the first is correct.
 */

export function cameraVideoOffsetS(
  offsets: Record<string, number> | undefined,
  cameraKey: string,
): number {
  const offset = offsets?.[cameraKey];
  return Number.isFinite(offset) ? (offset as number) : 0;
}

export function timelineTimeToVideoTime(
  offsets: Record<string, number> | undefined,
  cameraKey: string,
  timelineTimeS: number,
  videoWarmupS = 0,
): number {
  return Math.max(0, timelineTimeS - cameraVideoOffsetS(offsets, cameraKey) + videoWarmupS);
}

export function videoTimeToTimelineTime(
  offsets: Record<string, number> | undefined,
  cameraKey: string,
  videoTimeS: number,
  videoWarmupS = 0,
): number {
  return Math.max(0, videoTimeS + cameraVideoOffsetS(offsets, cameraKey) - videoWarmupS);
}

/** Where playback of this episode begins in its video file. */
export function episodeVideoStartS(
  offsets: Record<string, number> | undefined,
  cameraKey: string,
  videoWarmupS = 0,
): number {
  return timelineTimeToVideoTime(offsets, cameraKey, 0, videoWarmupS);
}
