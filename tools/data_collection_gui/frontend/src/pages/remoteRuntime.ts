import type { RolloutRuntimeOptions } from "../types";

/** The runtime options a rollout of a remote deployment sends.
 *
 * The policy runs on the GPU host and the R4 bridge drives it with RTC off and one action per
 * step, so the pi0.5 knobs on this page (RTC, draws, EMA, alignment, a servo pose left from an
 * earlier run) are not sent: hidden on the page, they must not reach the launcher either. The
 * task prompt is still the deployment's, and the grasp loop's own settings (trials, seed, arms,
 * insertion, recording) are the loop's, not the policy's, so they go through as for any
 * checkpoint. The page fills those only when the grasp loop is the selected mode.
 */
export function remoteRuntimeOptions(options: RolloutRuntimeOptions): RolloutRuntimeOptions {
  const graspLoop = Object.fromEntries(
    Object.entries(options).filter(([key, value]) => key.startsWith("graspLoop") && value !== undefined)
  ) as Partial<RolloutRuntimeOptions>;
  return { taskPrompt: options.taskPrompt, rtcMode: "disabled", ...graspLoop };
}
