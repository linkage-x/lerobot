import type { ProcessingItem, RecordedDataset, TrainingView } from "../types";

/** The prefix `rollout.py` gives a correction dataset (`DAGGER_PREFIX`), so the page can offer
 *  the recordings a DAgger run wrote without asking the operator to recognise them by eye.
 *  A convention, not a guarantee: the merge check is what actually decides, and an operator who
 *  renamed a dataset can still reach it through "show every recording". */
export const DAGGER_DATASET_PREFIX = "dagger_";

export function isDaggerDataset(dataset: RecordedDataset): boolean {
  return dataset.name.startsWith(DAGGER_DATASET_PREFIX);
}

const UNSAFE_VIEW_NAME_CHARS = /[^A-Za-z0-9._-]+/g;

/** Mirrors `_safe_training_view_name` in gateway.py. Kept in step so the name shown before the
 *  merge is the directory the merge writes -- an operator who has to discover the real name
 *  afterwards cannot tell a rename from an overwrite. */
export function safeTrainingViewName(name: string): string {
  const cleaned = name.replace(UNSAFE_VIEW_NAME_CHARS, "-").replace(/^[-.]+/, "").replace(/[-.]+$/, "");
  return cleaned || "training_view";
}

function lastPathSegment(value: string): string {
  const trimmed = value.replace(/\/+$/, "");
  return trimmed.slice(trimmed.lastIndexOf("/") + 1);
}

/** Mirrors `_merge_output_name` in gateway.py: the view a merge lands in when the operator
 *  names none. The single-source form carries the checkpoint step the corrections were
 *  collected against, because that is the one thing that distinguishes two merges of the same
 *  base view. */
export function mergedViewName(baseView: string, daggerRoots: string[], requested: string): string {
  if (requested.trim()) return safeTrainingViewName(requested.trim());
  let daggerTag = `plus${daggerRoots.length}dagger`;
  if (daggerRoots.length === 1) {
    const match = lastPathSegment(daggerRoots[0]).match(/_(\d{6}|\d{5,})$/);
    daggerTag = match ? `plus_dagger_${match[1]}` : "plus_dagger";
  }
  return safeTrainingViewName(`${lastPathSegment(baseView)}__${daggerTag}`);
}

export type EpisodeSelection = {
  /** Empty means "every episode of the base view", which is what the gateway reads an absent
   *  baseEpisodes as. A subset is never inferred -- it is only ever what was typed. */
  episodes: number[];
  error: string;
};

/** Parse "0-47, 50 52" into base-view episode indices.
 *
 *  Ranges are expanded here rather than sent as text because the gateway takes indices, and a
 *  holdout split is the whole reason this field exists: typing 48 numbers by hand to keep two
 *  episodes out of training is how a holdout ends up in the trained set.
 */
export function parseEpisodeSelection(text: string, totalEpisodes: number): EpisodeSelection {
  const empty: EpisodeSelection = { episodes: [], error: "" };
  const trimmed = text.trim();
  if (!trimmed) return empty;
  const seen = new Set<number>();
  for (const token of trimmed.split(/[\s,]+/).filter(Boolean)) {
    const range = token.match(/^(\d+)-(\d+)$/);
    const single = token.match(/^(\d+)$/);
    if (!range && !single) {
      return { episodes: [], error: `Cannot read ${token}; use episode numbers or ranges like 0-47` };
    }
    const start = Number(range ? range[1] : single![1]);
    const end = Number(range ? range[2] : single![1]);
    if (end < start) {
      return { episodes: [], error: `Range ${token} runs backwards` };
    }
    if (totalEpisodes > 0 && end >= totalEpisodes) {
      return {
        episodes: [],
        error: `Episode ${end} is outside the base view, which has ${totalEpisodes} episode(s), 0-${totalEpisodes - 1}`
      };
    }
    for (let episode = start; episode <= end; episode += 1) seen.add(episode);
  }
  return { episodes: Array.from(seen).sort((left, right) => left - right), error: "" };
}

export type DaggerCandidate = {
  dataset: RecordedDataset;
  qc: ProcessingItem | undefined;
  /** Why this recording cannot go into a merge, or "" when it can be ticked. */
  blockedReason: string;
};

/** The recordings offered as correction sources, newest first.
 *
 *  QC-warned recordings are blocked, unlike a training-view build which lets the operator
 *  acknowledge warnings: the merge script itself requires QC PASS on every DAgger root, so
 *  offering a warned one here would only produce a refusal after the click.
 */
export function daggerMergeCandidates(
  datasets: RecordedDataset[],
  processing: ProcessingItem[]
): DaggerCandidate[] {
  return datasets
    .filter((dataset) => dataset.datasetKind !== "training_view")
    .map((dataset) => {
      const qc = processing.find((item) => item.path === dataset.path);
      const blockedReason =
        dataset.totalEpisodes === 0
          ? "No episodes recorded"
          : qc?.status === "qc_pass"
            ? ""
            : qc?.status === "qc_warn"
              ? "QC raised warnings; a merge takes only QC PASS corrections"
              : qc?.status === "qc_failed"
                ? "QC failed — re-record before merging these corrections"
                : "Run QC in Dataset Processing before merging this dataset";
      return { dataset, qc, blockedReason };
    });
}

/** The candidates to draw, given the shortlist toggle.
 *
 *  A ticked recording stays visible even when it is outside the shortlist: turning the filter
 *  back on must not quietly drop a source from the merge that is about to run.
 */
export function visibleDaggerCandidates(
  candidates: DaggerCandidate[],
  options: { includeNonDagger: boolean; selectedPaths: string[] }
): DaggerCandidate[] {
  return candidates.filter(
    (candidate) =>
      options.includeNonDagger ||
      isDaggerDataset(candidate.dataset) ||
      options.selectedPaths.includes(candidate.dataset.path)
  );
}

export type MergeFormState = {
  baseView: string;
  daggerRoots: string[];
  baseEpisodes: number[];
  outputName: string;
};

/** What the check was run against, so a form edited afterwards cannot leave a stale "compatible"
 *  standing next to a Merge button. Deliberately excludes overwrite and copyVideos: neither
 *  changes what the check inspects, and invalidating on them would make ticking "replace" look
 *  like it had changed the answer. */
export function mergeFormFingerprint(form: MergeFormState): string {
  return JSON.stringify({
    baseView: form.baseView,
    daggerRoots: [...form.daggerRoots].sort(),
    baseEpisodes: [...form.baseEpisodes].sort((left, right) => left - right),
    outputName: form.outputName.trim()
  });
}

/** Why a merge cannot be started yet, in the order the operator would fix them, or "". */
export function mergeBlockedReason(input: {
  baseView: TrainingView | null;
  daggerCount: number;
  episodeError: string;
  keptBaseEpisodes: number;
  existingView: TrainingView | null;
  overwrite: boolean;
}): string {
  if (!input.baseView) return "Pick the training view these corrections were collected against";
  if (input.daggerCount === 0) return "Select at least one DAgger correction dataset";
  if (input.episodeError) return input.episodeError;
  if (input.keptBaseEpisodes === 0) return "The base episode list keeps nothing from the base view";
  if (input.existingView && !input.overwrite) {
    return `${input.existingView.name} already exists; tick Replace to overwrite it`;
  }
  return "";
}
