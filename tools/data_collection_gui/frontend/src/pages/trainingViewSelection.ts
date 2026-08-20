import type { CameraCropSpecs } from "../api";
import type { DatasetCameraFeature, RecordedDataset } from "../types";
import { fullFrameCrop, isFullFrame, type CropRect } from "./cropGeometry";

/** The task a recording belongs to, from its directory name.
 *
 *  `pick_and_place_20260819_171756` -> `pick_and_place`. Sessions of one task differ only by
 *  their capture timestamp, so the base name is what several recordings of the same task have
 *  in common. Mirrors `_dataset_task_base_name` in gateway.py; the two must agree because this
 *  side groups the checkboxes and that side names the directory they build into.
 */
export function taskBaseName(name: string): string {
  return name.match(/^(?<base>.+)_\d{8}_\d{6}(?:_\d{2})?$/)?.groups?.base ?? name;
}

/** The directory a build will write to, derived only from what goes into it.
 *
 *  Deterministic on purpose: rebuilding the same task lands on the same name and replaces it,
 *  which is what "this task gained a session, rebuild the view" should do. Mirrors
 *  `_training_view_name` in gateway.py so the page can name the output before the build starts.
 */
/**
 * The directory a build writes to. Mirrors the gateway's `_training_view_name`.
 *
 * Restricted to `[A-Za-z0-9._-]` because the name is also the training job name (the gateway
 * rejects anything outside that set) and the trailing half of the `local/<name>` repo id
 * (`validate_repo_id` rejects it on the same grounds). A name that only works as a directory
 * is a view that builds and then cannot be trained, which the page would have shown as ready.
 */
export function trainingViewName(datasetNames: string[], actionMode: string): string {
  if (datasetNames.length === 0) return "";
  if (datasetNames.length === 1) {
    // One source keeps its own full name, timestamp included: there is nothing to merge, and
    // collapsing it to the task name would make a single-session view claim the whole task.
    return safeViewName(`${datasetNames[0]}__${actionMode}`);
  }
  const bases = Array.from(new Set(datasetNames.map(taskBaseName))).sort();
  return safeViewName(`${bases.join("-")}__${actionMode}`);
}

function safeViewName(name: string): string {
  const cleaned = name.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[-.]+|[-.]+$/g, "");
  return cleaned || "training_view";
}

/** Why this recording cannot be built at `viewFps`, or "" when it can. */
export function viewFpsProblem(sourceFps: number | undefined, viewFps: number): string {
  if (!sourceFps || viewFps === 0) return "";
  if (sourceFps < viewFps) {
    return `recorded at ${sourceFps} fps — a ${viewFps} fps view would have to invent frames`;
  }
  if (sourceFps % viewFps !== 0) {
    // Only integer decimation exists. Nearest-frame resampling would make the gap between
    // kept frames alternate between 1 and 2 source frames, and the action is a per-frame
    // delta, so that swing lands directly in the values the policy learns.
    const divisors = [1, 2, 3, 4].filter((n) => sourceFps % n === 0).map((n) => sourceFps / n);
    return `${sourceFps} fps is not an integer multiple of ${viewFps} — pick ${divisors.join(", ")}`;
  }
  return "";
}

/** Why a whole selection cannot be built at `viewFps`, or "" when it can.
 *
 *  `viewFps === 0` means "keep the source rate", which only works when the sources already
 *  agree: merging 30 fps and 60 fps recordings without resampling puts two different per-frame
 *  action scales in one column, and nothing downstream can tell the halves apart.
 */
export function selectionFpsProblem(datasets: RecordedDataset[], viewFps: number): string {
  if (datasets.length === 0) return "";
  if (viewFps === 0) {
    const rates = Array.from(new Set(datasets.map((dataset) => dataset.fps ?? 0)));
    if (rates.length > 1) {
      return `the selection mixes ${rates.sort((a, b) => a - b).join(" and ")} fps — pick a rate they can all be decimated to`;
    }
    return "";
  }
  for (const dataset of datasets) {
    const problem = viewFpsProblem(dataset.fps, viewFps);
    if (problem) return `${dataset.name}: ${problem}`;
  }
  return "";
}

export type SelectionSummary = {
  datasets: number;
  episodes: number;
  excluded: number;
  /** Frames the view will hold, after per-source decimation to `viewFps`. */
  frames: number;
};

export function summarizeSelection(datasets: RecordedDataset[], viewFps: number): SelectionSummary {
  let episodes = 0;
  let excluded = 0;
  let frames = 0;
  for (const dataset of datasets) {
    const dropped = (dataset.excludedEpisodes ?? []).length;
    excluded += dropped;
    episodes += Math.max(0, dataset.totalEpisodes - dropped);
    const stride = viewFps > 0 && dataset.fps ? dataset.fps / viewFps : 1;
    // Scaled by the episodes that survive review, not the whole recording: the excluded ones
    // never reach the view and counting them would overstate the training set.
    const kept = dataset.totalEpisodes > 0 ? (dataset.totalEpisodes - dropped) / dataset.totalEpisodes : 0;
    frames += Math.round((dataset.totalFrames * kept) / (stride >= 1 ? stride : 1));
  }
  return { datasets: datasets.length, episodes, excluded, frames };
}

function cropForFeature(crops: Record<string, CropRect>, feature: DatasetCameraFeature): CropRect {
  return crops[feature.key] ?? fullFrameCrop(feature.width, feature.height);
}

/** The crop specs to send for a whole selection, or why it cannot be sent.
 *
 *  One box is drawn on one frame and then applied to every source in the build, because the
 *  exporter's crop is per camera key and global to the build. A source whose camera of the same
 *  name is a different size therefore has to be caught here, before the merge starts writing.
 */
export function cropSpecsForSelection(
  datasets: RecordedDataset[],
  enabled: boolean,
  crops: Record<string, CropRect>
): { crops?: CameraCropSpecs; error?: string; label?: string } {
  if (!enabled) return {};
  if (datasets.length === 0) return {};
  const specs: CameraCropSpecs = {};
  for (const dataset of datasets) {
    const features = (dataset.cameraFeatures ?? []).filter(
      (feature) => feature.width > 0 && feature.height > 0
    );
    if (features.length === 0) {
      return { error: `${dataset.name} has no camera metadata to crop against` };
    }
    for (const feature of features) {
      const rect = cropForFeature(crops, feature);
      const { x, y, w, h } = rect;
      if (![x, y, w, h].every(Number.isInteger)) {
        return { error: `${feature.key} crop must be integer pixels` };
      }
      if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > feature.width || y + h > feature.height) {
        return {
          error: `${feature.key} crop is outside ${feature.width}x${feature.height} in ${dataset.name}`
        };
      }
      if ([x, y, w, h].some((value) => value % 2 !== 0)) {
        return { error: `${feature.key} crop must use even x/y/w/h for H.264` };
      }
      if (!isFullFrame(rect, feature.width, feature.height)) {
        specs[feature.key] = [x, y, w, h];
      }
    }
  }
  const count = Object.keys(specs).length;
  return count > 0
    ? { crops: specs, label: `${count} camera crop${count === 1 ? "" : "s"}` }
    : { label: "full frame" };
}

export type TaskGroup = {
  base: string;
  datasets: RecordedDataset[];
};

/** Recordings grouped by the task they belong to, tasks ordered by their newest recording.
 *
 *  Grouping is the whole point of the list: the unit a view is built from is a task's sessions,
 *  and a flat list of 39 timestamped directories does not show which of them belong together.
 */
export function groupDatasetsByTask(datasets: RecordedDataset[]): TaskGroup[] {
  const groups = new Map<string, RecordedDataset[]>();
  for (const dataset of datasets) {
    const base = taskBaseName(dataset.name);
    const bucket = groups.get(base);
    if (bucket) bucket.push(dataset);
    else groups.set(base, [dataset]);
  }
  return Array.from(groups.entries())
    .map(([base, items]) => ({
      base,
      datasets: [...items].sort((a, b) => b.name.localeCompare(a.name))
    }))
    .sort((a, b) => b.datasets[0].name.localeCompare(a.datasets[0].name));
}
