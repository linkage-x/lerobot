/** Run states in which a training job is underway (the form is locked while one is). */
export const RUNNING_STATES = new Set(["syncing", "starting", "running"]);

/** The job whose settings the Training form should take over now, or null.
 *
 *  While a run is going the form is locked, and it used to keep showing this page's defaults
 *  (this machine, act, 20000 steps) beside a pi0.5 run on another host -- a screen that reads as
 *  "this is what is training" and is wrong in every field. So the form takes over the running
 *  job's settings, once per job: after that the operator's edits, once the run ends, are theirs.
 */
export function runJobToAdopt(
  run: { state: string; jobName: string } | null,
  adoptedJob: string
): string | null {
  if (!run || !RUNNING_STATES.has(run.state) || !run.jobName || run.jobName === adoptedJob) return null;
  return run.jobName;
}
