import type { RolloutRun } from "../types";

/**
 * Which of the session's controls may be pressed right now.
 *
 * Extracted from the page because these five buttons are the ones that move a robot, and they
 * are now rendered somewhere other than the card whose state they read: a bar that floats over
 * the page can be pressed from a scroll position where nothing else about the run is visible,
 * so the rule for each button has to be stated once, in one place, and be testable without a
 * browser. A bar that offered Start half a second before the runtime could accept it would look
 * like it worked and do nothing -- which is the failure this page has already been bitten by.
 */

/** The states in which a rollout process exists. Outside them there is no session to control. */
export const LIVE_STATES = new Set([
  "starting",
  "waiting",
  // The runtime between two activities: the end marker has printed, the gate has not been
  // reached. A live session, so End stays offered -- but not a pressable one, see `between`.
  "finishing",
  "homing",
  "resetting",
  "rolling"
]);

export type SessionAvailability = {
  canStart: boolean;
  canStop: boolean;
  canHome: boolean;
  canResetScene: boolean;
  canEnd: boolean;
};

/**
 * @param sceneResetRunnable  Whether the scene reset panel would run its own button: the mask is
 *   painted, motion is confirmed, and the panel is not disabled. The bar never decides this for
 *   itself -- it fires the panel's request, so it has to honour the panel's rule or it would send
 *   a reset the panel had refused.
 */
export function sessionAvailability(
  run: RolloutRun | null,
  busy: boolean,
  sceneResetRunnable: boolean
): SessionAvailability {
  const state = run?.state ?? "idle";
  const live = run !== null && LIVE_STATES.has(state);
  // Every command travels the same request; a second one sent while the first is in flight is
  // read against a run state that has already moved on.
  const idle = live && !busy;
  // `waiting` is the runtime printing `interactive_waiting_for_start`: everything before it --
  // homing, a minute of loading the policy, opening the cameras -- is `starting`, and a Start
  // pressed there is read by the listener thread and then cleared when the loop reaches its wait.
  //
  // `finishing` is the same trap at the other end of a rollout, and the one that actually bit:
  // the runtime prints `interactive_rollout_end` and then spends seconds writing a trace and
  // encoding a DAgger episode. The page used to call that `waiting` and offer all five buttons,
  // so a Reset scene pressed in the loop's own rhythm -- grade, reset, start -- was accepted by
  // the gateway, written to stdin, and then cleared at the gate. Nothing moved and nothing said
  // why. Only the gate marker may open these buttons.
  const between = idle && Boolean(run?.interactive) && state === "waiting";
  return {
    canStart: between,
    canStop: idle && Boolean(run?.interactive) && state === "rolling",
    canHome: between,
    canResetScene: between && sceneResetRunnable,
    // The one control that stays available while the arm is moving: it is the way out.
    canEnd: idle
  };
}

/** What the bar says about a state in which none of the arm controls are pressable.
 *
 * Empty whenever the state speaks for itself. The point is the two states an operator reads as
 * "the button is broken": `starting`, which is a policy load that can take a minute, and the two
 * in which the arm is already carrying out a command. */
export function sessionNote(run: RolloutRun | null): string {
  switch (run?.state) {
    case "starting":
      return "Loading the policy and opening the cameras — Start turns on when the runtime is waiting.";
    case "homing":
      return "Moving to the start pose.";
    case "resetting":
      return "Resetting the scene.";
    case "finishing":
      // The one an operator is most likely to read as a broken button, because the arm has
      // visibly stopped: the rollout is over and the runtime is still writing its correction
      // episode. Seconds, and longer the longer the takeover was.
      return "Rollout finished — writing the trace and any corrections. The controls come back when the runtime reaches its next command.";
    default:
      return "";
  }
}
