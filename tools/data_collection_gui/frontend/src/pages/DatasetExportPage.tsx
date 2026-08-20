import { useEffect, useMemo, useState } from "react";
import type { CameraCropSpecs, GuiSnapshot } from "../api";
import { api } from "../apiClient";
import type { DatasetCameraFeature, DatasetFramePreview, RecordedDataset, TrainingView } from "../types";
import {
  StatusDot,
  Metric,
  PageHeader,
  processingStatusLabel,
  qcWarnings,
  stateLabel,
  taskStatusDot
} from "../shared/ui";
import { CameraCropPicker, CropNumberField } from "./CameraCropPicker";
import { fullFrameCrop, isFullFrame, normalizeCrop, sideRoiCrop, type CropRect } from "./cropGeometry";
import {
  cropSpecsForSelection,
  groupDatasetsByTask,
  selectionFpsProblem,
  summarizeSelection,
  taskBaseName,
  trainingViewName,
  viewFpsProblem
} from "./trainingViewSelection";

type ActionMode = "absolute_ee" | "delta_ee_from_prev_cmd" | "delta_ee_from_current";

// Rates a training view may be built at. Mirrors TRAINING_VIEW_FPS_CHOICES in gateway.py.
// 0 means "keep whatever the recording is", which the exporter only allows when every source
// in one build already agrees.
const VIEW_FPS_CHOICES = [30, 15, 60, 0] as const;
const DEFAULT_VIEW_FPS = 30;

const actionModeCopy: Record<ActionMode, { label: string; blurb: string }> = {
  absolute_ee: {
    label: "Absolute EE",
    blurb:
      "action = absolute target pose (quaternion rotation). Rate-independent; the contract the recorder stores natively."
  },
  delta_ee_from_prev_cmd: {
    label: "Delta EE — vs previous command",
    blurb:
      "action = per-frame increment against the pose commanded on the previous frame (rotvec rotation). Arm tracking lag stays out of the action; a held frame is an exact zero."
  },
  delta_ee_from_current: {
    label: "Delta EE — vs measured pose",
    blurb:
      "action = per-frame increment against the measured pose. Purely reactive at deployment, but it bakes this rig's tracking lag into every action."
  }
};

function contractLabel(contract: string): string {
  return actionModeCopy[contract as ActionMode]?.label ?? contract;
}

/** Whether QC has cleared this recording for a build. A warned dataset is selectable: the
 *  warnings are the operator's call, and the gateway asks for confirmation before it starts. */
function selectableStatus(snapshot: GuiSnapshot, dataset: RecordedDataset): boolean {
  const status = snapshot.processing.find((item) => item.path === dataset.path)?.status;
  return status === "qc_pass" || status === "qc_warn";
}

function cropForFeature(crops: Record<string, CropRect>, feature: DatasetCameraFeature): CropRect {
  return crops[feature.key] ?? fullFrameCrop(feature.width, feature.height);
}

/** Trailing-edge debounce. The frame slider is dragged, and every distinct value it passes
 *  through would otherwise start an ffmpeg decode on the gateway. */
function useDebounced<T>(value: T, delayMs: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setSettled(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [value, delayMs]);
  return settled;
}

/** Workstation counterpart of Dataset Export: build the policy-ready view of a v3 recording. */
function TrainingViewPage({
  snapshot,
  busy,
  onBuildView,
  onOpenProcessing,
  onOpenReplay
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onBuildView: (
    paths: string[],
    actionMode?: string,
    cameraCrops?: CameraCropSpecs,
    viewFps?: number
  ) => void;
  onOpenProcessing: () => void;
  onOpenReplay: (path: string) => void;
}) {
  const exportStatus = snapshot.datasetExport;
  const building = exportStatus.state === "exporting";
  const [actionMode, setActionMode] = useState<ActionMode>("delta_ee_from_prev_cmd");
  const [viewFps, setViewFps] = useState<number>(DEFAULT_VIEW_FPS);
  const [cropEnabled, setCropEnabled] = useState(false);
  const [cameraCrops, setCameraCrops] = useState<Record<string, CropRect>>({});
  // Which recording the crop is drawn on. The box applies to every build this page starts, so
  // the preview picks a source rather than following the row that is about to be built.
  const [previewPath, setPreviewPath] = useState("");
  const [previewEpisode, setPreviewEpisode] = useState(0);
  const [previewFrame, setPreviewFrame] = useState(0);
  const [framePreview, setFramePreview] = useState<DatasetFramePreview | null>(null);
  // Which recordings go into the build. A view renumbers its episodes and computes
  // meta/stats.json over the whole set, so several sessions can only be combined at build time:
  // two views built separately share neither an episode index space nor a normalisation, and
  // there is no operation that appends a session to a finished view.
  const [selectedPaths, setSelectedPaths] = useState<string[]>([]);
  const [showEmptyRecordings, setShowEmptyRecordings] = useState(false);
  // Built views, for their manifests. The crop is baked into a view's video and this page keeps
  // nothing across a reload, so the manifest is the only record of what a build was made with.
  const [builtViews, setBuiltViews] = useState<TrainingView[]>([]);
  const allDatasets = snapshot.recordedDatasets ?? [];
  // Views are replay candidates, so they arrive in the same list as the recordings. They belong
  // under the recording they were derived from, not next to it as another build source.
  const datasets = allDatasets.filter((dataset) => dataset.datasetKind !== "training_view");
  // Derived, not stored: the dataset list is repolled every second and a stored path would
  // outlive the recording it names.
  const selected = datasets.filter((dataset) => selectedPaths.includes(dataset.path));
  const taskGroups = useMemo(() => groupDatasetsByTask(datasets), [datasets.map((d) => d.path).join("|")]);
  const cropCameraFeatures = Array.from(
    new Map(
      selected
        .flatMap((dataset) => dataset.cameraFeatures ?? [])
        .filter((feature) => feature.width > 0 && feature.height > 0)
        .map((feature) => [feature.key, feature] as const)
    ).values()
  );
  // Only the selected recordings: the box is drawn to decide what this build keeps, so drawing
  // it on a recording the build does not include answers a question nobody asked.
  const previewCandidates = selected.filter((dataset) =>
    (dataset.cameraFeatures ?? []).some((feature) => feature.width > 0 && feature.height > 0)
  );
  const summary = summarizeSelection(selected, viewFps);
  const fpsProblem = selectionFpsProblem(selected, viewFps);
  const cropResult = cropSpecsForSelection(selected, cropEnabled, cameraCrops);
  const targetViewName = trainingViewName(selected.map((dataset) => dataset.name), actionMode);
  const existingView = builtViews.find((view) => view.name === targetViewName) ?? null;
  const notQcReady = selected.filter((dataset) => {
    const status = snapshot.processing.find((item) => item.path === dataset.path)?.status;
    return status !== "qc_pass" && status !== "qc_warn";
  });
  const buildBlockedReason = selected.length === 0
    ? "Select at least one recording to build from"
    : notQcReady.length > 0
      ? `${notQcReady.map((dataset) => dataset.name).join(", ")} must pass QC first`
      : summary.episodes === 0
        ? "Every selected episode is marked not for training"
        : fpsProblem
          ? `Cannot build at ${viewFps === 0 ? "the source rate" : `${viewFps} fps`}: ${fpsProblem}`
          : cropResult.error ?? "";

  const toggleSelected = (path: string) => {
    setSelectedPaths((current) =>
      current.includes(path) ? current.filter((item) => item !== path) : [...current, path]
    );
  };
  const setGroupSelected = (group: RecordedDataset[], on: boolean) => {
    const paths = group.map((dataset) => dataset.path);
    setSelectedPaths((current) =>
      on ? Array.from(new Set([...current, ...paths])) : current.filter((item) => !paths.includes(item))
    );
  };
  /** Adopt a built view's settings. The manifest is what a build actually used, so reusing it
   *  cannot drift from the frames a policy was trained on the way a retyped box would. */
  const reuseViewSettings = (view: TrainingView) => {
    if (view.actionMode) setActionMode(view.actionMode as ActionMode);
    setViewFps(view.fps || DEFAULT_VIEW_FPS);
    const crops = Object.entries(view.cameraCrops ?? {});
    setCropEnabled(crops.length > 0);
    setCameraCrops((current) => {
      const next = { ...current };
      for (const [key, box] of crops) {
        const [x, y, w, h] = box;
        next[key] = { x, y, w, h };
      }
      return next;
    });
    // The sources only when nothing is ticked yet. Adopting a view's settings is usually the
    // first step of "build that task again, now that it has more sessions", so replacing an
    // explicit selection would drop the very recording that prompted the rebuild.
    if (selectedPaths.length === 0) {
      const roots = new Set(view.sourceRoots ?? []);
      const stillPresent = datasets.filter((dataset) => roots.has(dataset.path)).map((d) => d.path);
      if (stillPresent.length > 0) setSelectedPaths(stillPresent);
    }
  };
  // Derived, not stored: the dataset list is repolled every second and a stored path would
  // survive the recording it names being deleted.
  const activePreviewPath = previewCandidates.some((dataset) => dataset.path === previewPath)
    ? previewPath
    : previewCandidates[0]?.path ?? "";
  const previewEpisodes = framePreview?.episodes ?? [];
  const activePreviewEpisode = previewEpisodes.some((item) => item.episode === previewEpisode)
    ? previewEpisode
    : previewEpisodes[0]?.episode ?? 0;
  const previewEpisodeFrames =
    previewEpisodes.find((item) => item.episode === activePreviewEpisode)?.frames ?? 0;
  const maxPreviewFrame = Math.max(0, previewEpisodeFrames - 1);
  const activePreviewFrame = Math.min(previewFrame, maxPreviewFrame);
  // The gateway decodes on demand, so the URL only follows the slider once it settles.
  const settledPreviewFrame = useDebounced(activePreviewFrame, 150);

  // Refetched whenever a build finishes: the view that just landed is the one whose settings
  // the operator is most likely to reuse next.
  useEffect(() => {
    let cancelled = false;
    void api.fetchTrainingViews().then((result) => {
      if (!cancelled) setBuiltViews(result);
    });
    return () => {
      cancelled = true;
    };
  }, [exportStatus.state]);

  useEffect(() => {
    if (!cropEnabled || !activePreviewPath) {
      setFramePreview(null);
      return;
    }
    let cancelled = false;
    api.fetchDatasetFramePreview(activePreviewPath).then((result) => {
      if (cancelled) return;
      setFramePreview(result);
      setPreviewFrame(0);
    });
    return () => {
      cancelled = true;
    };
  }, [cropEnabled, activePreviewPath]);

  const setCameraCrop = (key: string, rect: CropRect) => {
    setCameraCrops((current) => ({ ...current, [key]: rect }));
  };
  const useRecommendedCameraCrop = () => {
    setCropEnabled(true);
    setCameraCrops(
      Object.fromEntries(
        cropCameraFeatures.map((feature) => [
          feature.key,
          sideRoiCrop(feature.key, feature.width, feature.height)
        ])
      )
    );
  };
  const resetCameraCrops = () => {
    setCameraCrops(
      Object.fromEntries(
        cropCameraFeatures.map((feature) => [feature.key, fullFrameCrop(feature.width, feature.height)])
      )
    );
  };
  // The gateway refuses to build a view of a dataset that has not passed QC, so the row has to
  // say where a dataset stands before the button is pressed. Shown rather than filtered: on this
  // profile every recording is a build candidate, and hiding the ones that need QC is what made
  // the Thor page look like it had silently lost datasets.
  const qcItemFor = (dataset: RecordedDataset) =>
    snapshot.processing.find((item) => item.path === dataset.path);
  const viewsFor = (dataset: RecordedDataset) =>
    allDatasets.filter(
      (candidate) =>
        candidate.datasetKind === "training_view" &&
        (candidate.viewOf
          ? candidate.viewOf === dataset.path
          : candidate.viewOfName === dataset.name || candidate.name.startsWith(`${dataset.name}__`))
    );

  return (
    <div className="page-stack">
      <PageHeader
        title="Training View"
        subtitle="re-express a recorded v3 dataset in the action contract the policy will be trained on"
      />
      <section className="panel">
        <div className="panel-heading">
          <h2>Action Contract</h2>
          <span>{actionModeCopy[actionMode].label}</span>
        </div>
        <p className="panel-note">
          Recording always stores absolute EE. The delta contracts are derived here by differencing
          consecutive dataset frames — a delta computed during capture would span one control tick
          (200 Hz) instead of one frame (30 Hz) and drive the arm ~6.7&times; too slow. Videos are
          symlinked unless crop is enabled, so a full-frame view costs almost no disk.
        </p>
        <div className="mujoco-mode-picker" role="group" aria-label="Action contract">
          {(Object.keys(actionModeCopy) as ActionMode[]).map((mode) => (
            <button
              key={mode}
              className={actionMode === mode ? "active" : ""}
              disabled={busy || building}
              onClick={() => setActionMode(mode)}
              type="button"
            >
              {actionModeCopy[mode].label}
            </button>
          ))}
        </div>
        <p className="panel-note">{actionModeCopy[actionMode].blurb}</p>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>Frame Rate</h2>
          <span>{viewFps === 0 ? "keep source rate" : `${viewFps} fps`}</span>
        </div>
        <p className="panel-note">
          The action is a <em>per-frame</em> delta, so the same real motion is twice as large per
          frame at 30 fps as at 60. Views built at different rates therefore cannot be merged —
          the difference lands in the action values themselves and nothing downstream can see it.
          Building every view at one rate is what lets a 60 fps session join the 30 fps baseline.
          Frames are dropped, never interpolated, and the videos are not re-encoded.
        </p>
        <div className="mujoco-mode-picker" role="group" aria-label="Training view frame rate">
          {VIEW_FPS_CHOICES.map((choice) => (
            <button
              key={choice}
              className={viewFps === choice ? "active" : ""}
              disabled={busy || building}
              onClick={() => setViewFps(choice)}
              type="button"
            >
              {choice === 0 ? "Source rate" : `${choice} fps`}
            </button>
          ))}
        </div>
      </section>

      <section className="panel camera-crop-panel">
        <div className="panel-heading">
          <h2>Camera Crop</h2>
          <span>{cropEnabled ? "enabled" : "full frame"}</span>
        </div>
        <p className="panel-note">
          Crop is applied only to the generated training view. The raw recording stays unchanged.
          The box is drawn on a real frame of a recording because that is the only thing that
          answers the question being asked — whether the workspace still fits.
        </p>
        <div className="control-row">
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={cropEnabled}
              disabled={busy || building || cropCameraFeatures.length === 0}
              onChange={(event) => setCropEnabled(event.target.checked)}
            />
            <span>Use crop for training view</span>
          </label>
          <button disabled={busy || building || cropCameraFeatures.length === 0} onClick={useRecommendedCameraCrop}>
            Use Side ROI
          </button>
          <button disabled={busy || building || cropCameraFeatures.length === 0} onClick={resetCameraCrops}>
            Full Frame
          </button>
        </div>
        {cropEnabled && cropCameraFeatures.length > 0 ? (
          <>
            <div className="crop-source-row">
              <label>
                <span>Frame from</span>
                <select
                  value={activePreviewPath}
                  disabled={busy || building || previewCandidates.length === 0}
                  onChange={(event) => {
                    setPreviewPath(event.target.value);
                    setPreviewEpisode(0);
                    setPreviewFrame(0);
                  }}
                >
                  {previewCandidates.map((dataset) => (
                    <option key={dataset.path} value={dataset.path}>
                      {dataset.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Episode</span>
                <select
                  value={activePreviewEpisode}
                  disabled={busy || building || previewEpisodes.length === 0}
                  onChange={(event) => {
                    setPreviewEpisode(Number(event.target.value));
                    setPreviewFrame(0);
                  }}
                >
                  {previewEpisodes.map((item) => (
                    <option key={item.episode} value={item.episode}>
                      {item.episode} ({item.frames} fr)
                    </option>
                  ))}
                </select>
              </label>
              <label className="crop-frame-scrub">
                <span>
                  Frame {activePreviewFrame}
                  {maxPreviewFrame > 0 ? ` / ${maxPreviewFrame}` : ""}
                </span>
                <input
                  type="range"
                  min={0}
                  max={maxPreviewFrame}
                  value={activePreviewFrame}
                  disabled={busy || building || maxPreviewFrame === 0}
                  onChange={(event) => setPreviewFrame(Number(event.target.value))}
                />
              </label>
            </div>
            <div className="camera-crop-grid">
              {cropCameraFeatures.map((feature) => {
                const rect = cropForFeature(cameraCrops, feature);
                const editable = !busy && !building;
                return (
                  <article className="camera-crop-card" key={feature.key}>
                    <header>
                      <div>
                        <strong>{feature.key}</strong>
                        <small>
                          {feature.width}x{feature.height} source
                          {isFullFrame(rect, feature.width, feature.height) ? " · full frame" : ""}
                        </small>
                      </div>
                      <div className="crop-card-actions">
                        <button
                          disabled={!editable}
                          onClick={() => setCameraCrop(feature.key, sideRoiCrop(feature.key, feature.width, feature.height))}
                        >
                          Side ROI
                        </button>
                        <button
                          disabled={!editable}
                          onClick={() => setCameraCrop(feature.key, fullFrameCrop(feature.width, feature.height))}
                        >
                          Full
                        </button>
                      </div>
                    </header>
                    <CameraCropPicker
                      frameUrl={
                        activePreviewPath
                          ? api.datasetFrameUrl(
                              activePreviewPath,
                              feature.key,
                              activePreviewEpisode,
                              settledPreviewFrame
                            )
                          : ""
                      }
                      frameW={feature.width}
                      frameH={feature.height}
                      rect={rect}
                      disabled={!editable}
                      onChange={(next) => setCameraCrop(feature.key, next)}
                    />
                    <div className="crop-fields">
                      {(["x", "y", "w", "h"] as const).map((field) => (
                        <CropNumberField
                          key={field}
                          label={field}
                          value={rect[field]}
                          disabled={!editable}
                          onCommit={(value) =>
                            setCameraCrop(
                              feature.key,
                              normalizeCrop({ ...rect, [field]: value }, feature.width, feature.height)
                            )
                          }
                        />
                      ))}
                    </div>
                  </article>
                );
              })}
            </div>
          </>
        ) : null}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Sources</h2>
          <span>
            {selected.length} of {datasets.length} selected
          </span>
        </div>
        <p className="panel-note">
          Pick every recording that should train one policy. They are merged here because this is
          the only moment they can be: a view renumbers its episodes and computes its
          normalisation over the whole set, so two views built separately cannot be combined
          afterwards, and adding a session later means rebuilding from all of them at once.
        </p>
        {datasets.length === 0 ? (
          <div className="empty-dataset-list">No recorded datasets yet. Record an episode first.</div>
        ) : (
          <div className="source-group-list">
            {taskGroups.map((group) => {
              const visible = group.datasets.filter(
                (dataset) => showEmptyRecordings || dataset.totalEpisodes > 0
              );
              const hidden = group.datasets.length - visible.length;
              if (visible.length === 0 && hidden > 0 && !showEmptyRecordings) {
                return (
                  <section className="source-group" key={group.base}>
                    <header>
                      <strong>{group.base}</strong>
                      <span>{hidden} empty recording(s)</span>
                    </header>
                  </section>
                );
              }
              const selectable = visible.filter(
                (dataset) => selectableStatus(snapshot, dataset) && dataset.totalEpisodes > 0
              );
              const allOn =
                selectable.length > 0 && selectable.every((dataset) => selectedPaths.includes(dataset.path));
              return (
                <section className="source-group" key={group.base}>
                  <header>
                    <strong>{group.base}</strong>
                    <span>
                      {group.datasets.length} session(s) ·{" "}
                      {group.datasets.reduce((total, dataset) => total + dataset.totalEpisodes, 0)} episode(s)
                    </span>
                    <button
                      type="button"
                      disabled={busy || building || selectable.length === 0}
                      onClick={() => setGroupSelected(selectable, !allOn)}
                    >
                      {allOn ? "Clear task" : "Select task"}
                    </button>
                  </header>
                  {visible.map((dataset) => {
                    const excluded = dataset.excludedEpisodes ?? [];
                    const kept = Math.max(0, dataset.totalEpisodes - excluded.length);
                    const qc = qcItemFor(dataset);
                    const qcStatus = qc?.status;
                    const warned = qcStatus === "qc_warn";
                    const qcReady = qcStatus === "qc_pass" || warned;
                    const warnings = qc ? qcWarnings(qc) : [];
                    const rowFpsProblem = viewFpsProblem(dataset.fps, viewFps);
                    const checked = selectedPaths.includes(dataset.path);
                    const views = viewsFor(dataset);
                    const blockedReason = !qcReady
                      ? qcStatus === "qc_failed"
                        ? "QC failed — fix or re-record before training on this"
                        : "Run QC in Dataset Processing before building a view"
                      : dataset.totalEpisodes === 0
                        ? "No episodes recorded"
                        : kept === 0
                          ? "Every episode is marked not for training"
                          : "";
                    return (
                      <div className={checked ? "source-row selected" : "source-row"} key={dataset.path}>
                        <label className="source-row-main">
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={busy || building || Boolean(blockedReason)}
                            onChange={() => toggleSelected(dataset.path)}
                          />
                          <div>
                            <div className="row-title">
                              <StatusDot
                                state={
                                  qcStatus === "qc_pass"
                                    ? "running"
                                    : warned
                                      ? "warning"
                                      : qcStatus === "qc_failed"
                                        ? "error"
                                        : "idle"
                                }
                              />
                              <strong>{dataset.name}</strong>
                              <em>{qcStatus ? processingStatusLabel[qcStatus] : "QC not run"}</em>
                            </div>
                            <p>
                              {kept}
                              {excluded.length > 0 ? ` of ${dataset.totalEpisodes}` : ""} episode(s) ·{" "}
                              {dataset.totalFrames} frames
                              {dataset.fps ? ` · ${dataset.fps} fps` : ""}
                            </p>
                            {blockedReason ? (
                              // Stated on the row rather than left to the error the build would
                              // have returned: QC is a page away, and "why can I not tick this"
                              // has to be answerable without pressing anything.
                              <p className="panel-note">
                                {blockedReason}. {qc?.qcSummary ?? "QC has not run on this recording."}
                              </p>
                            ) : null}
                            {checked && rowFpsProblem ? (
                              <p className="panel-note">Cannot build at {viewFps} fps: {rowFpsProblem}.</p>
                            ) : null}
                            {checked && warned ? (
                              <p className="panel-note">
                                {warnings.length ? warnings.join(" · ") : qc?.message} — building asks for
                                confirmation first.
                              </p>
                            ) : null}
                            {checked && excluded.length > 0 ? (
                              // Shown before the build: this is the operator's own review deciding
                              // what reaches training, and it changes what the button does.
                              <p className="panel-note">
                                Episode{excluded.length > 1 ? "s" : ""} {excluded.join(", ")} marked not
                                for training in Episode Replay.
                              </p>
                            ) : null}
                          </div>
                        </label>
                        {views.length > 0 ? (
                          <div className="view-list">
                            {views.map((view) => (
                              <div className="view-row" key={view.path}>
                                <div>
                                  <strong>{contractLabel(view.actionContract ?? "")}</strong>
                                  <p>
                                    {view.totalEpisodes} episode(s) · {view.totalFrames} frames · built{" "}
                                    {view.updatedAt}
                                  </p>
                                  <p className="view-path">{view.path}</p>
                                </div>
                                <button disabled={busy || building} onClick={() => onOpenReplay(view.path)}>
                                  Open in Replay
                                </button>
                              </div>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                  {hidden > 0 && showEmptyRecordings === false ? (
                    <p className="panel-note">{hidden} empty recording(s) hidden.</p>
                  ) : null}
                </section>
              );
            })}
          </div>
        )}
        <div className="control-row">
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={showEmptyRecordings}
              onChange={(event) => setShowEmptyRecordings(event.target.checked)}
            />
            <span>Show recordings with no episodes</span>
          </label>
          <button disabled={busy} onClick={onOpenProcessing}>
            Open Processing
          </button>
        </div>
      </section>

      <section className="panel build-panel">
        <div className="panel-heading">
          <h2>Build</h2>
          <span>{targetViewName || "nothing selected"}</span>
        </div>
        {builtViews.length > 0 ? (
          <div className="control-row">
            <label>
              <span>Reuse settings from</span>
              <select
                value=""
                disabled={busy || building}
                onChange={(event) => {
                  const view = builtViews.find((item) => item.name === event.target.value);
                  if (view) reuseViewSettings(view);
                }}
              >
                <option value="">a view built earlier…</option>
                {builtViews.map((view) => (
                  <option key={view.name} value={view.name}>
                    {view.name} ({view.episodes} ep
                    {Object.keys(view.cameraCrops ?? {}).length > 0 ? ", cropped" : ""})
                  </option>
                ))}
              </select>
            </label>
          </div>
        ) : null}
        <p className="panel-note">
          A view's settings are recorded in its manifest, which is the only place they survive —
          the crop is baked into the view's video and this page keeps nothing across a reload.
          Adopting them keeps whatever is ticked, so the usual rebuild is: select the task, then
          adopt the crop and rate its last view used.
        </p>
        <div className="summary-grid">
          <Metric label="Sources" value={summary.datasets} />
          <Metric label="Episodes" value={summary.episodes} />
          <Metric label="Excluded" value={summary.excluded} />
          <Metric label="Frames (est.)" value={summary.frames.toLocaleString()} />
          <Metric label="Contract" value={actionModeCopy[actionMode].label} />
          <Metric label="Rate" value={viewFps === 0 ? "source" : `${viewFps} fps`} />
          <Metric label="Crop" value={cropEnabled ? (cropResult.label ?? "—") : "full frame"} />
        </div>
        {existingView ? (
          // The build writes to a fixed name, so a rebuild replaces whatever is there. Said
          // before the button, because a checkpoint trained on the old contents keeps pointing
          // at this path and nothing else would announce that its frames had changed.
          <p className="panel-note">
            {targetViewName} already exists ({existingView.episodes} episode(s), built{" "}
            {existingView.buildId || existingView.modifiedAt}) and will be replaced. Checkpoints
            trained from it keep pointing at this path.
          </p>
        ) : null}
        {buildBlockedReason ? <p className="panel-note">{buildBlockedReason}.</p> : null}
        <div className="control-row">
          <button
            className="primary"
            disabled={busy || building || Boolean(buildBlockedReason)}
            title={buildBlockedReason || undefined}
            onClick={() => onBuildView(selected.map((dataset) => dataset.path), actionMode, cropResult.crops, viewFps)}
          >
            {building ? "Building…" : `Build View from ${selected.length} recording(s)`}
          </button>
        </div>
      </section>

      {exportStatus.state !== "idle" && (
        <section className="panel">
          <div className="panel-heading">
            <h2>Build Status</h2>
            <span className="state-pill">
              <StatusDot state={exportStatus.state} />
              {stateLabel(exportStatus.state)}
            </span>
          </div>
          <div className="summary-grid">
            <Metric
              label="Sources"
              value={(exportStatus.datasetRoots ?? [exportStatus.datasetRoot])
                .filter(Boolean)
                .map((root) => root.split("/").pop())
                .join(", ") || "—"}
            />
            <Metric label="Contract" value={contractLabel(exportStatus.target)} />
            <Metric label="View" value={exportStatus.outputPath || "—"} />
            <Metric label="Episodes" value={exportStatus.selectedEpisodes} />
            <Metric label="Frames" value={exportStatus.totalFrames} />
            <Metric label="Latest log" value={exportStatus.message || "—"} />
          </div>
        </section>
      )}
    </div>
  );
}

export function DatasetExportPage({
  snapshot,
  busy,
  onExportTask,
  onExportApprovedDataset,
  onOpenProcessing,
  onOpenReplay
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onExportTask: (id: string) => void;
  onExportApprovedDataset: (
    paths: string[],
    actionMode?: string,
    cameraCrops?: CameraCropSpecs,
    viewFps?: number
  ) => void;
  onOpenProcessing: () => void;
  onOpenReplay: (path: string) => void;
}) {
  const exportStatus = snapshot.datasetExport;
  // A warned dataset is exportable, with the warnings acknowledged. Leaving it out of this list
  // is what made a single warn look like "QC pending" and silently withdraw the dataset.
  const eligible = snapshot.processing.filter(
    (item) => item.status === "qc_pass" || item.status === "qc_warn"
  );
  const hasEligible = eligible.length > 0;
  const exportableTasks = (snapshot.tasks ?? []).filter((t) => t.datasetRepoId);
  const exporting = exportStatus.state === "exporting";

  // The FR3 workstation recorder already writes LeRobot v3, so there is no raw->v3 export to
  // run here (that is the Thor GMSL2 path). What it needs instead is a training view: the same
  // episodes with the action column in whichever contract the policy will be trained on.
  if (snapshot.deployment?.profile === "workstation") {
    return (
      <TrainingViewPage
        snapshot={snapshot}
        busy={busy}
        onBuildView={onExportApprovedDataset}
        onOpenProcessing={onOpenProcessing}
        onOpenReplay={onOpenReplay}
      />
    );
  }

  return (
    <div className="page-stack">
      <PageHeader title="Dataset Export" subtitle="consolidate task sessions or QC-approved datasets into LeRobot v3" />
      <section className="panel">
        <div className="panel-heading">
          <h2>Consolidate a Task</h2>
          <span>{exportableTasks.length} exportable</span>
        </div>
        <p className="panel-note">
          Merge every recorded session of a task into one LeRobot v3 dataset under the exports root. Raw sessions are left untouched; re-run any time.
        </p>
        {exportableTasks.length === 0 ? (
          <div className="empty-dataset-list">No tasks with a dataset repo id. Create one in Task Library first.</div>
        ) : (
          <div className="processing-list">
            {exportableTasks.map((task) => (
              <div className="processing-row" key={task.id}>
                <div className="processing-row-main static">
                  <div>
                    <div className="row-title">
                      <StatusDot state={taskStatusDot[task.status]} />
                      <strong>{task.name}</strong>
                      <em>{task.datasetRepoId}</em>
                    </div>
                    <p>{task.completedEpisodes} episode(s) recorded across its sessions</p>
                  </div>
                  <div className="processing-stats">
                    <button
                      disabled={busy || exporting}
                      onClick={() => onExportTask(task.id)}
                    >
                      {exporting && exportStatus.taskId === task.id ? "Exporting…" : "Export v3"}
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        {exportStatus.state !== "idle" && (
          <div className="summary-grid">
            <Metric label="Export state" value={stateLabel(exportStatus.state)} />
            <Metric label="Output" value={exportStatus.outputPath || "—"} />
            <Metric label="Episodes" value={exportStatus.selectedEpisodes} />
            <Metric label="Frames" value={exportStatus.totalFrames} />
            <Metric label="Message" value={exportStatus.message} />
          </div>
        )}
        {exportStatus.outputPath ? (
          <div className="control-row">
            <button disabled={busy || exporting} onClick={() => onOpenReplay(exportStatus.outputPath)}>Open Replay</button>
          </div>
        ) : null}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Approved Datasets</h2>
          <span>{eligible.length} ready</span>
        </div>
        {hasEligible ? (
          <div className="processing-list">
            {eligible.map((item) => {
              const warnings = qcWarnings(item);
              const warned = item.status === "qc_warn";
              return (
                <div className="processing-row" key={item.path}>
                  <div className="processing-row-main static">
                    <div>
                      <div className="row-title">
                        <StatusDot state={warned ? "warning" : "running"} />
                        <strong>{item.name}</strong>
                        {item.trajectoryVersion ? <em>{item.trajectoryVersion}</em> : null}
                      </div>
                      <p>{item.qcSummary}</p>
                      {warned ? (
                        <p className="panel-note">
                          {warnings.length ? warnings.join(" · ") : item.message} — exporting asks for
                          confirmation first.
                        </p>
                      ) : null}
                    </div>
                    <div className="processing-stats">
                      <span>{item.totalEpisodes} ep · {item.totalFrames} fr</span>
                      <small>{item.updatedAt}</small>
                      <button
                        disabled={busy || exporting}
                        onClick={() => onExportApprovedDataset([item.path])}
                      >
                        {exporting && exportStatus.datasetRoot === item.path ? "Exporting…" : "Export v3"}
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="empty-dataset-list">
            No QC-passed datasets yet. Run QC in Dataset Processing first.
            <div className="control-row">
              <button disabled={busy} onClick={onOpenProcessing}>Open Processing</button>
            </div>
          </div>
        )}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Export Plan</h2>
          <span className="state-pill">
            <StatusDot state={exportStatus.state} />
            {stateLabel(exportStatus.state)}
          </span>
        </div>
        <div className="summary-grid">
          <Metric label="Target" value={exportStatus.target} />
          <Metric label="Dataset root" value={exportStatus.datasetRoot} />
          <Metric label="Output" value={exportStatus.outputPath} />
          <Metric label="Episodes" value={exportStatus.selectedEpisodes} />
          <Metric label="Frames" value={exportStatus.totalFrames} />
          <Metric label="Message" value={exportStatus.message} />
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Layer Manifest</h2>
          <span>Raw / Debug / Training</span>
        </div>
        <div className="layer-grid">
          {[
            ["Raw Layer", exportStatus.includeRaw, "video, robot state, controller state, sidecar jsonl"],
            ["Debug Layer", exportStatus.includeDebug, "MCAP, Rerun log, timeline index"],
            ["Training Layer", exportStatus.includeTraining, "LeRobot v3, Parquet, dataset card"]
          ].map(([label, enabled, detail]) => (
            <div className="layer-card" key={String(label)}>
              <strong>{label}</strong>
              <span>{enabled ? "included" : "excluded"}</span>
              <p>{detail}</p>
            </div>
          ))}
        </div>
        <ul className="manifest-list">
          {exportStatus.manifest.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}

