import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../apiClient";
import type { Checkpoint, CheckpointListing, TrainingHost } from "../types";

/**
 * The list of trained checkpoints, on this machine or on a training host.
 *
 * Used in two shapes. On the Training page it is a manager: sizes, disk reclamation, and the
 * fetch that brings a remotely-trained checkpoint to the machine the robot is wired to. On the
 * Rollout page it is a picker.
 *
 * Either way it leads with the verdict rather than the path, because the question an operator
 * is actually asking of this list is "which of these may I run", and on this rig the answer is
 * not visible from the filename.
 */

export function formatBytes(bytes: number): string {
  if (!bytes) return "—";
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  return `${Math.round(bytes / 1e6)} MB`;
}

export function successRate(checkpoint: Checkpoint): string {
  const outcomes = checkpoint.outcomes;
  if (!outcomes) return "—";
  // Aborted runs are excluded from the denominator: they were ended for reasons that say
  // nothing about the policy, and counting them would drag every rate toward zero.
  const graded = outcomes.success + outcomes.failure;
  if (graded === 0) return `0/0 (${outcomes.aborted} aborted)`;
  return `${outcomes.success}/${graded}`;
}

const VERDICT_LABEL: Record<Checkpoint["verdict"], string> = {
  ok: "Matches rig",
  warn: "Check first",
  block: "Cannot roll out"
};

export function verdictClass(verdict: Checkpoint["verdict"]): string {
  return verdict === "ok" ? "ok" : verdict === "warn" ? "warn" : "error";
}

/** Free-text match over the fields an operator would actually recall about a run. */
export function checkpointMatches(checkpoint: Checkpoint, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return [
    checkpoint.id,
    checkpoint.jobName,
    checkpoint.policyType,
    checkpoint.datasetRepoId,
    checkpoint.view.actionMode ?? "",
    ...checkpoint.cameras
  ]
    .join(" ")
    .toLowerCase()
    .includes(needle);
}

type Props = {
  /** "picker" hides the destructive actions and reports the selection upward. */
  mode: "manage" | "picker";
  selectedId?: string;
  onSelect?: (checkpoint: Checkpoint | null) => void;
  /** Bump to force a refresh from outside (e.g. after a training run finishes). */
  refreshToken?: number;
  disabled?: boolean;
};

export function CheckpointBrowser({
  mode,
  selectedId = "",
  onSelect,
  refreshToken = 0,
  disabled = false
}: Props) {
  const [hosts, setHosts] = useState<TrainingHost[]>([]);
  const [hostId, setHostId] = useState("local");
  const [listing, setListing] = useState<CheckpointListing | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState("");
  const [hideBlocked, setHideBlocked] = useState(false);
  const [expanded, setExpanded] = useState<string>("");
  const [selection, setSelection] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const refresh = useCallback(async (id: string) => {
    setLoading(true);
    setError("");
    const payload = await api.fetchCheckpoints(id);
    setListing(payload);
    if (payload && !payload.ok) setError(payload.error || "Could not list checkpoints.");
    setLoading(false);
  }, []);

  useEffect(() => {
    void api.fetchTrainingHosts().then(setHosts);
  }, []);

  useEffect(() => {
    void refresh(hostId);
  }, [hostId, refresh, refreshToken]);

  const checkpoints = useMemo(() => {
    const all = listing?.checkpoints ?? [];
    return all.filter(
      (item) => checkpointMatches(item, query) && (!hideBlocked || item.verdict !== "block")
    );
  }, [listing, query, hideBlocked]);

  const totalBytes = useMemo(
    // `last` is a symlink onto a numbered step, so counting it would double the newest
    // checkpoint in a figure whose whole purpose is deciding what to delete.
    () =>
      (listing?.checkpoints ?? [])
        .filter((item) => !item.aliasOf)
        .reduce((sum, item) => sum + item.sizeBytes, 0),
    [listing]
  );

  const isRemote = listing?.host?.kind === "remote";
  // Delete reaches the gateway's own repo, so it is offered only for the machine the gateway
  // runs on. A remote host's checkpoints are managed by fetching them here first.
  const canBulkDelete = mode === "manage" && !isRemote;
  // `last` is a symlink onto a numbered step: the backend refuses it, and letting it be ticked
  // would put a row in the batch that can only ever come back as a failure.
  const deletable = useMemo(() => checkpoints.filter((item) => !item.aliasOf), [checkpoints]);
  const selectedBytes = useMemo(
    () =>
      deletable
        .filter((item) => selection.includes(item.id))
        .reduce((sum, item) => sum + item.sizeBytes, 0),
    [deletable, selection]
  );
  const deletableBytes = useMemo(
    () => deletable.reduce((sum, item) => sum + item.sizeBytes, 0),
    [deletable]
  );

  // A tick is a claim about a row that is on screen. Switching machine, typing in the search
  // box, or refreshing after a delete can retire any of them, and an id left behind would keep
  // "Delete 4 selected" pointing at something the operator can no longer see.
  useEffect(() => {
    const visible = new Set(deletable.map((item) => item.id));
    setSelection((current) => {
      const next = current.filter((id) => visible.has(id));
      return next.length === current.length ? current : next;
    });
  }, [deletable]);

  const toggleOne = (id: string) =>
    setSelection((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id]
    );

  const allSelected = deletable.length > 0 && selection.length === deletable.length;

  /** Every checkpoint of a job except its newest -- the batch this page is actually for.
   *
   * Keyed on the highest step per job rather than on mtime: a checkpoint fetched from a
   * training host carries the mtime of the copy, which would nominate the wrong survivor.
   */
  const selectSuperseded = () => {
    const newestStep = new Map<string, number>();
    for (const item of deletable) {
      newestStep.set(item.jobName, Math.max(newestStep.get(item.jobName) ?? -1, item.step));
    }
    setSelection(
      deletable
        .filter((item) => item.step < (newestStep.get(item.jobName) ?? item.step))
        .map((item) => item.id)
    );
  };

  const onFetch = async (checkpoint: Checkpoint) => {
    setBusy(true);
    setError("");
    setNotice("");
    const result = await api.fetchCheckpointToLocal(hostId, checkpoint.id);
    setBusy(false);
    if (!result.ok) {
      setError(result.error || "Fetch failed.");
      return;
    }
    setNotice(result.message || "Checkpoint fetched.");
  };

  const onDelete = async (checkpoint: Checkpoint) => {
    if (!window.confirm(`Delete ${checkpoint.id}? This frees ${formatBytes(checkpoint.sizeBytes)}.`)) {
      return;
    }
    setBusy(true);
    setError("");
    setNotice("");
    const result = await api.deleteCheckpoint(checkpoint.id);
    setBusy(false);
    if (!result.ok) {
      setError(result.error || "Delete failed.");
      return;
    }
    setNotice(result.message || "Checkpoint deleted.");
    if (selectedId === checkpoint.id) onSelect?.(null);
    await refresh(hostId);
  };

  const onDeleteSelected = async () => {
    const targets = deletable.filter((item) => selection.includes(item.id));
    if (targets.length === 0) return;
    const bytes = targets.reduce((sum, item) => sum + item.sizeBytes, 0);
    if (
      !window.confirm(
        `Delete ${targets.length} checkpoint(s)? This frees ${formatBytes(bytes)} and cannot be undone.`
      )
    ) {
      return;
    }
    setBusy(true);
    setError("");
    setNotice("");
    const result = await api.deleteCheckpoints(targets.map((item) => item.id));
    setBusy(false);
    const failed = result.failed ?? [];
    // A batch reports per checkpoint, so both halves are shown: the space actually reclaimed,
    // and the ids that survived. Collapsing it to one banner would hide whichever came second.
    if (failed.length > 0) {
      setError(failed.map((item) => `${item.checkpointId}: ${item.error}`).join(" · "));
    } else if (!result.ok) {
      setError(result.error || "Bulk delete failed.");
    }
    const deleted = result.deleted ?? [];
    if (deleted.length > 0) {
      setNotice(result.message || `Deleted ${deleted.length} checkpoint(s).`);
      if (selectedId && deleted.includes(selectedId)) onSelect?.(null);
    }
    setSelection([]);
    await refresh(hostId);
  };

  return (
    <div className="checkpoint-browser">
      <div className="row-actions checkpoint-toolbar">
        <label className="field inline">
          <span>Machine</span>
          <select
            value={hostId}
            onChange={(event) => setHostId(event.target.value)}
            disabled={disabled || loading}
          >
            {hosts.map((host) => (
              <option key={host.id} value={host.id}>
                {host.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field inline grow">
          <span>Search</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="job, policy, dataset, camera…"
          />
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={hideBlocked}
            onChange={(event) => setHideBlocked(event.target.checked)}
          />
          <span>Only ones I can run</span>
        </label>
        <button type="button" onClick={() => void refresh(hostId)} disabled={loading || disabled}>
          {loading ? "Scanning…" : "Refresh"}
        </button>
      </div>

      {error && <div className="banner banner-error">{error}</div>}
      {notice && <div className="banner banner-ok">{notice}</div>}

      {canBulkDelete && deletable.length > 0 && (
        <div className="row-actions checkpoint-bulk">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={allSelected}
              ref={(node) => {
                if (node) node.indeterminate = selection.length > 0 && !allSelected;
              }}
              disabled={disabled || busy}
              onChange={() => setSelection(allSelected ? [] : deletable.map((item) => item.id))}
            />
            <span>Select all {deletable.length} shown</span>
          </label>
          <button type="button" onClick={selectSuperseded} disabled={disabled || busy}>
            Select superseded
          </button>
          <button
            type="button"
            onClick={() => setSelection([])}
            disabled={disabled || busy || selection.length === 0}
          >
            Clear
          </button>
          <button
            type="button"
            className="danger"
            disabled={disabled || busy || selection.length === 0}
            onClick={() => void onDeleteSelected()}
          >
            {busy && selection.length > 0
              ? "Deleting…"
              : selection.length === 0
                ? "Delete selected"
                : `Delete ${selection.length} · ${formatBytes(selectedBytes)}`}
          </button>
          <span className="bulk-summary">
            {selection.length === 0
              ? `${deletable.length} deletable · ${formatBytes(deletableBytes)}`
              : `${selection.length} of ${deletable.length} ticked`}
          </span>
        </div>
      )}

      {listing?.rig && (
        <p className="hint">
          This rig: robot <code>{listing.rig.robotIp || "unset"}</code>, tool frame{" "}
          <code>{listing.rig.targetFrameName}</code>, cameras{" "}
          <code>{listing.rig.cameraKeys.join(", ") || "none"}</code>. A checkpoint is judged
          against these.
        </p>
      )}

      {!loading && checkpoints.length === 0 && (
        <p className="hint">
          {listing?.checkpoints?.length
            ? "No checkpoint matches this filter."
            : "No checkpoints on this machine yet. Train one on the Training page."}
        </p>
      )}

      {checkpoints.length > 0 && (
        <table className="table checkpoint-table">
          <thead>
            <tr>
              {(mode === "picker" || canBulkDelete) && <th aria-label="select" />}
              <th>Checkpoint</th>
              <th>Policy</th>
              <th>Dataset</th>
              <th>Size</th>
              <th>Rollouts</th>
              <th>Verdict</th>
              <th aria-label="actions" />
            </tr>
          </thead>
          <tbody>
            {checkpoints.map((checkpoint) => {
              const isSelected = selectedId === checkpoint.id;
              const isExpanded = expanded === checkpoint.id;
              return [
                <tr
                  key={checkpoint.id}
                  className={isSelected ? "selected-row" : undefined}
                  onClick={
                    mode === "picker" && !disabled ? () => onSelect?.(checkpoint) : undefined
                  }
                >
                  {mode === "picker" && (
                    <td>
                      <input
                        type="radio"
                        name="checkpoint"
                        checked={isSelected}
                        disabled={disabled}
                        onChange={() => onSelect?.(checkpoint)}
                      />
                    </td>
                  )}
                  {canBulkDelete && (
                    <td>
                      {!checkpoint.aliasOf && (
                        <input
                          type="checkbox"
                          aria-label={`select ${checkpoint.id}`}
                          checked={selection.includes(checkpoint.id)}
                          disabled={disabled || busy}
                          onClick={(event) => event.stopPropagation()}
                          onChange={() => toggleOne(checkpoint.id)}
                        />
                      )}
                    </td>
                  )}
                  <td>
                    <strong>{checkpoint.jobName}</strong>
                    <div className="cell-sub">
                      step {checkpoint.step.toLocaleString()}
                      {checkpoint.aliasOf ? ` · last → ${checkpoint.aliasOf}` : ""}
                      {checkpoint.totalSteps
                        ? ` of ${checkpoint.totalSteps.toLocaleString()}`
                        : ""}
                    </div>
                  </td>
                  <td>{checkpoint.policyType || "—"}</td>
                  <td>
                    <div className="cell-sub">{checkpoint.view.actionMode || "unknown contract"}</div>
                    <div className="cell-sub">
                      {checkpoint.view.exists
                        ? `${checkpoint.view.episodes} ep · ${checkpoint.view.fps} fps`
                        : "view not on this machine"}
                    </div>
                  </td>
                  <td>{formatBytes(checkpoint.sizeBytes)}</td>
                  <td>{successRate(checkpoint)}</td>
                  <td>
                    <span className={`pill pill-${verdictClass(checkpoint.verdict)}`}>
                      {VERDICT_LABEL[checkpoint.verdict]}
                    </span>
                  </td>
                  <td className="row-actions">
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        setExpanded(isExpanded ? "" : checkpoint.id);
                      }}
                    >
                      {isExpanded ? "Hide" : "Details"}
                    </button>
                    {mode === "manage" && isRemote && (
                      <button
                        type="button"
                        disabled={busy || disabled}
                        onClick={(event) => {
                          event.stopPropagation();
                          void onFetch(checkpoint);
                        }}
                      >
                        Fetch here
                      </button>
                    )}
                    {mode === "manage" && !isRemote && !checkpoint.aliasOf && (
                      <button
                        type="button"
                        className="danger"
                        disabled={busy || disabled}
                        onClick={(event) => {
                          event.stopPropagation();
                          void onDelete(checkpoint);
                        }}
                      >
                        Delete
                      </button>
                    )}
                  </td>
                </tr>,
                isExpanded ? (
                  <tr key={`${checkpoint.id}-detail`} className="detail-row">
                    <td colSpan={mode === "picker" || canBulkDelete ? 8 : 7}>
                      <CheckpointDetail checkpoint={checkpoint} />
                    </td>
                  </tr>
                ) : null
              ];
            })}
          </tbody>
        </table>
      )}

      {mode === "manage" && (listing?.checkpoints?.length ?? 0) > 0 && (
        <p className="hint">
          {listing?.checkpoints.filter((item) => !item.aliasOf).length} checkpoint(s) using{" "}
          {formatBytes(totalBytes)} on {listing?.host.label}.{" "}
          {isRemote
            ? "Fetch one here before rolling it out — the robot and its cameras are on this machine."
            : "Intermediate checkpoints can be deleted once a later one is proven better — “Select superseded” ticks every step of each job except its newest."}
        </p>
      )}
    </div>
  );
}

function CheckpointDetail({ checkpoint }: { checkpoint: Checkpoint }) {
  const safety = checkpoint.contract.safety;
  return (
    <div className="checkpoint-detail">
      {checkpoint.issues.length > 0 && (
        <ul className="issue-list">
          {checkpoint.issues.map((issue) => (
            <li key={`${issue.field}-${issue.level}`} className={`issue issue-${verdictClass(issue.level as Checkpoint["verdict"])}`}>
              <strong>{issue.field}</strong>: {issue.message}
            </li>
          ))}
        </ul>
      )}
      <dl className="detail-grid">
        <div>
          <dt>Tool frame</dt>
          <dd>{checkpoint.contract.targetFrameName || "unrecorded"}</dd>
        </div>
        <div>
          <dt>Robot</dt>
          <dd>{checkpoint.contract.robotIp || "unrecorded"}</dd>
        </div>
        <div>
          <dt>Cameras</dt>
          <dd>{checkpoint.cameras.join(", ") || "—"}</dd>
        </div>
        <div>
          <dt>Camera config</dt>
          <dd>{checkpoint.contract.cameraConfig || "unrecorded"}</dd>
        </div>
        <div>
          <dt>Action chunk</dt>
          <dd>
            {checkpoint.chunkSize ?? "—"} / {checkpoint.nActionSteps ?? "—"} executed
          </dd>
        </div>
        <div>
          <dt>Safety envelope</dt>
          <dd>
            {safety
              ? `first ${safety.firstFrameMaxPosDeltaMm ?? "?"} mm / ${safety.firstFrameMaxRotDeltaDeg ?? "?"}°, ` +
                `step ${safety.maxStepPosDeltaMm ?? "?"} mm / ${safety.maxStepRotDeltaDeg ?? "?"}°, ` +
                `leash ${safety.maxLeashPosDeltaMm ?? "?"} mm / ${safety.maxLeashRotDeltaDeg ?? "?"}°`
              : "unrecorded"}
          </dd>
        </div>
        <div className="wide">
          <dt>Dataset</dt>
          <dd>
            <code>{checkpoint.datasetRepoId || "—"}</code>
            <div className="cell-sub">{checkpoint.datasetRoot}</div>
            {checkpoint.view.relocated && (
              <div className="cell-sub">
                Matched by name in this repo. The checkpoint itself records{" "}
                <code>{checkpoint.recordedDatasetRoot}</code>, which is the path on the machine
                that trained it.
              </div>
            )}
          </dd>
        </div>
        <div className="wide">
          <dt>Weights</dt>
          <dd>
            <code>{checkpoint.path}</code>
          </dd>
        </div>
      </dl>
    </div>
  );
}
