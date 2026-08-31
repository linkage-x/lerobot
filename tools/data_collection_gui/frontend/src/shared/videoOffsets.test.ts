import { describe, expect, it } from "vitest";
import {
  cameraVideoOffsetS,
  episodeVideoStartS,
  timelineTimeToVideoTime,
  videoTimeToTimelineTime,
} from "./videoOffsets";

const EE = "observation.images.ee";

describe("cameraVideoOffsetS", () => {
  it("is zero for a camera the gateway reported no offset for", () => {
    expect(cameraVideoOffsetS({}, EE)).toBe(0);
    expect(cameraVideoOffsetS(undefined, EE)).toBe(0);
  });

  it("is zero for a non-finite entry rather than poisoning every seek with NaN", () => {
    expect(cameraVideoOffsetS({ [EE]: Number.NaN }, EE)).toBe(0);
    expect(cameraVideoOffsetS({ [EE]: Number.POSITIVE_INFINITY }, EE)).toBe(0);
  });
});

describe("a v3 chunk that packs every episode into one file", () => {
  // Episode 29 of the merged insert view: 20 episodes in one 315 s mp4, this one starting at
  // 297.583 s. The gateway reports the file's own zero as -297.583 on the episode's time axis.
  const offsets = { [EE]: -297.583 };

  it("starts playback at the episode's own place in the file, not at the file's start", () => {
    expect(episodeVideoStartS(offsets, EE)).toBeCloseTo(297.583, 6);
  });

  it("maps a timeline time to the matching time in the packed file", () => {
    expect(timelineTimeToVideoTime(offsets, EE, 0)).toBeCloseTo(297.583, 6);
    expect(timelineTimeToVideoTime(offsets, EE, 17.4667)).toBeCloseTo(315.0497, 4);
  });

  it("maps back, so the frame counter follows the episode and not the file", () => {
    expect(videoTimeToTimelineTime(offsets, EE, 297.583)).toBeCloseTo(0, 6);
    expect(videoTimeToTimelineTime(offsets, EE, 302.583)).toBeCloseTo(5, 6);
  });

  it("round-trips", () => {
    const t = 12.5;
    expect(videoTimeToTimelineTime(offsets, EE, timelineTimeToVideoTime(offsets, EE, t))).toBeCloseTo(t, 6);
  });
});

describe("a GMSL2 recording whose camera file starts after the shared t0", () => {
  // The sign that was already in use: a positive offset means the file's zero sits *later* on
  // the timeline, so the seek target is earlier than the timeline time.
  const offsets = { cam_00: 0.1 };

  it("seeks behind the timeline time by the camera's own start delay", () => {
    expect(timelineTimeToVideoTime(offsets, "cam_00", 1.0)).toBeCloseTo(0.9, 6);
    expect(videoTimeToTimelineTime(offsets, "cam_00", 0.9)).toBeCloseTo(1.0, 6);
  });

  it("never asks for a negative currentTime", () => {
    expect(timelineTimeToVideoTime(offsets, "cam_00", 0)).toBe(0);
  });
});

describe("videoWarmupS", () => {
  it("shifts the seek forward and is undone on the way back", () => {
    expect(timelineTimeToVideoTime({}, EE, 2, 0.5)).toBeCloseTo(2.5, 6);
    expect(videoTimeToTimelineTime({}, EE, 2.5, 0.5)).toBeCloseTo(2, 6);
  });
});
