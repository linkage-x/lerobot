// The Training page's form: defaults, what the numbers mean in time, and the request it sends.
// Pure functions so they can be tested without a gateway (trainingForm.test.ts).

export type Crop = [number, number, number, number];

export type CameraChoice = { name: string; crop: Crop | null };

export type TrainingForm = {
  exports: string[];
  hand: "right" | "left";
  cameras: CameraChoice[];
  steps: number;
  batchSize: number;
  horizon: number;
  nActionSteps: number;
  frameStride: number;
  history: number;
  imageLongSide: number;
  saveFreq: number;
  numWorkers: number;
  wrench: boolean;
  imageAug: boolean;
  jobName: string;
  hostId: string;
};

// Starting values, not tuned for a task. 60 fps exports; stride 2 = 30 Hz action steps; a 32-step chunk
// is about a second of motion; half of it is executed before the policy is asked again.
export const DEFAULT_FORM: TrainingForm = {
  exports: [],
  hand: "right",
  cameras: [{ name: "cam_10", crop: null }],
  steps: 20000,
  batchSize: 32,
  horizon: 32,
  nActionSteps: 16,
  frameStride: 2,
  history: 2,
  imageLongSide: 320,
  saveFreq: 2000,
  numWorkers: 8,
  wrench: false,
  imageAug: true,
  jobName: "",
  hostId: "local"
};

export function stepHz(fps: number, stride: number): number {
  return fps > 0 && stride > 0 ? fps / stride : 0;
}

export function chunkSeconds(fps: number, stride: number, horizon: number): number {
  const hz = stepHz(fps, stride);
  return hz > 0 ? horizon / hz : 0;
}

/** Output size of a camera after its crop: aspect kept, long side `longSide`, both even (as the trainer). */
export function outputSize(crop: Crop, longSide: number): [number, number] {
  const w = crop[2] - crop[0];
  const h = crop[3] - crop[1];
  const scale = longSide / Math.max(w, h);
  return [Math.max(2, Math.round((w * scale) / 2) * 2), Math.max(2, Math.round((h * scale) / 2) * 2)];
}

/** What is wrong with a crop for a `size[0]` x `size[1]` image, or "" when it is fine. */
export function cropProblem(crop: Crop, size: [number, number]): string {
  const [x0, y0, x1, y1] = crop;
  if (![x0, y0, x1, y1].every((v) => Number.isInteger(v))) return "whole pixels only";
  if (x0 < 0 || y0 < 0 || x1 > size[0] || y1 > size[1]) return `inside 0..${size[0]} x 0..${size[1]}`;
  if (x1 - x0 < 64 || y1 - y0 < 64) return "at least 64 px each way";
  return "";
}

export function formProblems(form: TrainingForm, cameraSizes: Record<string, [number, number]>): string[] {
  const problems: string[] = [];
  if (form.exports.length === 0) return ["Pick at least one export."];
  if (!form.cameras.some((c) => c.name === "cam_10")) problems.push("cam_10 (on the gripper) is required.");
  const names = form.cameras.map((c) => c.name);
  if (new Set(names).size !== names.length) problems.push("A camera is listed twice.");
  for (const cam of form.cameras) {
    const size = cameraSizes[cam.name];
    if (!size) {
      problems.push(`${cam.name} is not in every chosen export.`);
    } else if (cam.crop) {
      const problem = cropProblem(cam.crop, size);
      if (problem) problems.push(`${cam.name} crop: ${problem}.`);
    }
  }
  if (form.nActionSteps > form.horizon) problems.push("Steps executed per call cannot exceed the chunk length.");
  if (form.jobName && !/^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$/.test(form.jobName)) {
    problems.push("Run name: letters, digits, . _ - only.");
  }
  return problems;
}

/** The body of POST /api/training/start. */
export function startRequest(form: TrainingForm): Record<string, unknown> {
  return {
    ...form,
    jobName: form.jobName.trim(),
    cameras: form.cameras.map((c) => ({ name: c.name, crop: c.crop })),
    policy: "act"
  };
}

export function progressFraction(step: number, total: number): number {
  return total > 0 ? Math.min(1, Math.max(0, step / total)) : 0;
}
