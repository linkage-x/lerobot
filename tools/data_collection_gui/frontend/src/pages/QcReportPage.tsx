import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GuiSnapshot } from "../api";
import type { BoxPreviewPayload, BoxCaliLog, BoxCaliLogLine, CollectionTask, ConfigSummary, DeviceStatus, EpisodeAnnotation, EventLogItem, ProcessingItem, ProcessingStatus, RecordedDataset, RecordingStatus, ReplayStatus, SubtaskSegment, TaskStatus, DatasetExportStatus, AnnotationOutcome, AnnotationQuality, ReviewStatus } from "../types";
import { StatusDot, Metric, PageHeader, stateLabel, QualityOverview, processingStatusLabel, datasetNamePrefixes, taskDatasetBaseName, processingItemsForTask, taskNeedsQcExportConfirmation } from "../shared/ui";

export function QcReportPage({ snapshot }: { snapshot: GuiSnapshot }) {
  return (
    <div className="page-stack">
      <PageHeader title="QC Report" subtitle="episode-level data quality checks for sync, completeness, device health, and schema readiness" />
      <QualityOverview snapshot={snapshot} />
      <section className="panel">
        <div className="panel-heading">
          <h2>Checks</h2>
          <span>minimum viable QC</span>
        </div>
        <div className="check-table">
          {[
            ["camera fps", `${snapshot.configSummary.fps} fps target`, "pass"],
            ["frame drop", `${snapshot.trajectory.filter((point) => point.event === "gap").length} gaps`, "review"],
            ["timestamp gap", `${Math.max(0, ...snapshot.trajectory.map((point) => point.skewMs)).toFixed(1)} ms max`, "review"],
            ["action latency", `${snapshot.replay.trackingErrorMm.toFixed(1)} mm replay error`, "pass"],
            ["LeRobot schema", snapshot.replay.dataStatus === "loaded" ? "trajectory loaded" : "pending dataset", "review"]
          ].map(([name, value, state]) => (
            <div className="check-row" key={name}>
              <strong>{name}</strong>
              <span>{value}</span>
              <em>{state}</em>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

