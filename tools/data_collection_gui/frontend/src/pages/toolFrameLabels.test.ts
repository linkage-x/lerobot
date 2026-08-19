import { describe, expect, it } from "vitest";

import episodeReplayPageSource from "./EpisodeReplayPage.tsx?raw";
import teleoperationPageSource from "./TeleoperationPage.tsx?raw";

/**
 * The two FR3 tool frames are 410.85 mm apart on the same URDF and share an orientation, so
 * replaying a dataset recorded in the other one does not fail -- the arm puts the fingertips where
 * the other frame's origin was and runs to completion. The label an operator reads before pressing
 * Run is therefore load-bearing, and it has to come from the snapshot (`targetFrameName`, which the
 * gateway fills from `robot.target_frame_name`) rather than from a string that was true when it was
 * typed. Both pages used to spell `pika_task_tcp` out, and both were correct when they did.
 *
 * `pika_gripper_ee` is deliberately not banned: it is also a `RealEndEffectorMode`, which selects
 * the gripper stack for a replay rather than describing the IK frame, and that use is typed.
 */
const PAGES: [string, string][] = [
  ["EpisodeReplayPage.tsx", episodeReplayPageSource],
  ["TeleoperationPage.tsx", teleoperationPageSource]
];

describe("tool frame labels", () => {
  for (const [page, source] of PAGES) {
    it(`${page} does not spell the recording frame out`, () => {
      expect(
        source.includes("pika_task_tcp"),
        `${page} hardcodes pika_task_tcp. Read the frame off the snapshot instead: a label that ` +
          "disagrees with robot.target_frame_name is worse than no label, because the replay it " +
          "describes still completes -- 411 mm from where the operator expects."
      ).toBe(false);
    });

    it(`${page} takes the frame it shows from the snapshot`, () => {
      expect(
        source.includes("targetFrameName"),
        `${page} no longer reads targetFrameName. If it stopped showing a tool frame at all this ` +
          "check can go; if it still shows one, it is showing a constant."
      ).toBe(true);
    });
  }
});
