import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GuiSnapshot } from "../api";
import type { BoxPreviewPayload, BoxCaliLog, BoxCaliLogLine, CollectionTask, ConfigSummary, DeviceStatus, EpisodeAnnotation, EventLogItem, ProcessingItem, ProcessingStatus, RecordedDataset, RecordingStatus, ReplayStatus, SubtaskSegment, TaskStatus, DatasetExportStatus, AnnotationOutcome, AnnotationQuality, ReviewStatus } from "../types";
import { StatusDot, Metric, PageHeader, stateLabel, QualityOverview, processingStatusLabel, datasetNamePrefixes, taskDatasetBaseName, processingItemsForTask, taskNeedsQcExportConfirmation } from "../shared/ui";

export function PlaceholderPage({ title }: { title: string }) {
  return (
    <div className="page-stack">
      <PageHeader title={title} subtitle="placeholder page reserved for the data factory workflow" />
      <section className="panel placeholder-panel">
        <h2>{title}</h2>
        <p>This page is intentionally a placeholder in this pass.</p>
      </section>
    </div>
  );
}

