import { useEffect, useRef, useState } from "react";
import { api } from "../apiClient";
import type { GuiSnapshot } from "../api";
import type { HardwareDocument, HardwareField, HardwareInstance, HardwareKind, HardwareModel, HardwarePayload } from "../hardwareTypes";
import { changeRobot, changeTool, instanceFor, isHardwareDocument, nextCameraName, parseField, withParameter } from "./hardware";

type InstanceProps = {
  title: string;
  kind: HardwareKind;
  instance: HardwareInstance | null;
  models: HardwareModel[];
  onChange: (instance: HardwareInstance | null, model?: HardwareModel) => void;
  onError: (field: string, error: string) => void;
  prefix: string;
};

function ParameterField({ field, value, onChange, onError }: {
  field: HardwareField;
  value: unknown;
  onChange: (value: unknown) => void;
  onError: (error: string) => void;
}) {
  const format = (next: unknown) => next == null ? "" : field.kind === "json" ? JSON.stringify(next) : String(next);
  const [text, setText] = useState(() => format(value));
  const [error, setError] = useState("");
  const lastValue = useRef(value);
  useEffect(() => {
    if (value !== lastValue.current) {
      setText(format(value));
      lastValue.current = value;
    }
  }, [value]);
  function edit(next: string) {
    setText(next);
    try {
      const parsed = parseField(field, next);
      lastValue.current = parsed;
      onChange(parsed);
      setError("");
      onError("");
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setError(message);
      onError(message);
    }
  }
  return (
    <label className="annotation-field">
      <span>{field.label}{field.required ? " *" : ""}</span>
      {field.kind === "boolean" ? (
        <select value={value == null ? "" : String(value)} onChange={(e) => onChange(e.target.value === "" ? null : e.target.value === "true")}>
          {field.nullable && <option value="">Default</option>}
          <option value="true">Yes</option><option value="false">No</option>
        </select>
      ) : field.kind === "select" ? (
        <select value={value == null ? "" : String(value)} onChange={(e) => {
          const selected = field.options?.find((v) => String(v) === e.target.value);
          onChange(selected ?? null);
        }}>
          {(field.nullable || value == null) && <option value="">Default</option>}
          {field.options?.map((option) => <option key={String(option)} value={String(option)}>{String(option)}</option>)}
        </select>
      ) : field.kind === "json" ? (
        <textarea rows={2} value={text} onChange={(e) => edit(e.target.value)} placeholder={field.nullable ? "Default" : "JSON"} aria-invalid={!!error} />
      ) : (
        <input value={text} onChange={(e) => edit(e.target.value)} inputMode={field.kind === "integer" || field.kind === "number" ? "decimal" : "text"} aria-invalid={!!error} />
      )}
      {error && <small className="hardware-field-error">{error}</small>}
    </label>
  );
}

function InstanceForm({ title, kind, instance, models, onChange, onError, prefix }: InstanceProps) {
  const candidates = models.filter((model) => model.kind === kind);
  const selected = candidates.find((model) => model.type === instance?.type);
  function fields(advanced: boolean) {
    return <div className="hardware-fields">{selected?.fields.filter((field) => field.advanced === advanced).map((field) => (
      <ParameterField key={`${instance?.type}:${field.name}`} field={field} value={instance?.params[field.name]}
        onChange={(value) => instance && onChange(withParameter(instance, field.name, value))}
        onError={(error) => onError(`${prefix}.${field.name}`, error)} />
    ))}</div>;
  }
  return (
    <section className="hardware-instance">
      <label className="annotation-field hardware-model-select">
        <span>{title}</span>
        <select value={instance?.type ?? ""} onChange={(e) => {
          const model = candidates.find((candidate) => candidate.type === e.target.value);
          onChange(model ? instanceFor(model) : null, model);
        }}>
          <option value="">{kind === "end_effector" ? "Integrated / no separate end effector" : "Select a model"}</option>
          {instance && !selected && <option value={instance.type}>{instance.type} — adapter not registered</option>}
          {candidates.map((model) => <option key={model.type} value={model.type} disabled={!model.available}>{model.label}{model.available ? "" : " — configuration dependency missing"}</option>)}
        </select>
      </label>
      {selected?.description && <p className="panel-note">{selected.description}</p>}
      {selected && <>
        {fields(false)}
        {selected.fields.some((f) => f.advanced) && <details className="hardware-advanced"><summary>Advanced parameters</summary>{fields(true)}</details>}
      </>}
      {instance && !selected && <p className="panel-note error">Install this hardware's adapter on the gateway to edit or apply it.</p>}
    </section>
  );
}

const mountingFields: HardwareField[] = [
  { name: "urdf_path", label: "Robot + tool model (URDF)", kind: "string" },
  { name: "target_frame_name", label: "TCP frame", kind: "string" },
  { name: "start_joint_positions", label: "Home joints (rad)", kind: "json" },
  { name: "workspace_min", label: "Workspace minimum [x, y, z] (m)", kind: "json" },
  { name: "workspace_max", label: "Workspace maximum [x, y, z] (m)", kind: "json" },
].map((field) => ({ ...field, kind: field.kind as HardwareField["kind"], required: false, nullable: true, advanced: true, default: null }));

export function HardwareConfiguration({ snapshot }: { snapshot: GuiSnapshot }) {
  const [payload, setPayload] = useState<HardwarePayload | null>(null);
  const [draft, setDraft] = useState<HardwareDocument | null>(null);
  const [revision, setRevision] = useState<string | null>(null);
  const [editorKey, setEditorKey] = useState(0);
  const [dirty, setDirty] = useState(false);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [runtimeYaml, setRuntimeYaml] = useState("");
  const [loading, setLoading] = useState(true);
  const importRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    setLoading(true);
    const next = await api.fetchHardware();
    if (next) {
      setPayload(next);
      replace(next.document, next.revision);
      setWarnings(next.warnings);
      setError("");
    } else {
      setError("Hardware configuration is unavailable. Connect to the Python gateway; configuration changes cannot be saved in offline preview.");
    }
    setLoading(false);
  }
  useEffect(() => { void refresh(); }, []);

  function replace(document: HardwareDocument, nextRevision: string | null) {
    setDraft(structuredClone(document));
    setRevision(nextRevision);
    setDirty(false);
    setFieldErrors({});
    setRuntimeYaml("");
    setEditorKey((key) => key + 1);
  }
  function update(next: HardwareDocument) {
    setDraft(next);
    setDirty(true);
    setMessage("");
    setWarnings([]);
    setRuntimeYaml("");
  }
  function fieldError(field: string, reason: string) {
    setFieldErrors((previous) => {
      const next = { ...previous };
      if (reason) next[field] = reason;
      else delete next[field];
      return next;
    });
  }
  async function operation(action: () => Promise<void>) {
    setWorking(true);
    setError("");
    setMessage("");
    try { await action(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setWorking(false); }
  }
  async function save(apply: boolean) {
    if (!draft || !payload) return;
    const saved = await api.saveHardware(draft, revision);
    if (!saved.ok) throw new Error(saved.error ?? "Save failed");
    setDraft(saved.document);
    setRevision(saved.revision);
    setDirty(false);
    setWarnings(saved.warnings);
    setPayload({ ...payload, setups: saved.setups });
    if (apply) {
      const applied = await api.applyHardware(saved.document.id, saved.revision);
      if (!applied.ok) throw new Error(applied.error ?? "Apply failed; your draft is saved.");
      setPayload({ ...payload, ...applied, models: payload.models });
      setMessage("Configuration applied. New sessions use these devices; this selection is restored after gateway restart.");
    } else {
      setMessage("Draft saved. Apply it to change the devices used by new sessions.");
    }
  }

  if (!payload || !draft) return <section className="panel">
    <h2>Hardware configuration</h2>
    <p className={error ? "panel-note error" : "panel-note"}>{loading ? "Loading hardware models…" : error}</p>
    {!loading && <button onClick={() => void refresh()}>Retry</button>}
  </section>;

  const busy = ["connecting", "armed", "recording", "review", "saving", "discarding"].includes(snapshot.recording.state)
    || ["starting", "running"].includes(snapshot.teleop.state);
  const invalid = Object.values(fieldErrors).some(Boolean);
  const models = payload.models;
  const toolModels = models.filter((m) => m.kind !== "end_effector" || m.compatibleRobots.includes(draft.robot?.type ?? ""));

  return <section className="panel hardware-config" aria-label="Hardware configuration">
    <div className="panel-heading"><h2>Hardware configuration</h2><span>{dirty ? "Unsaved changes" : "Saved"}</span></div>
    <p className="panel-note">Choose models and configure each device here. Moving a setup only requires updating its connection parameters when mounting geometry stays the same.</p>
    <div className="hardware-toolbar">
      <label className="annotation-field"><span>Saved configurations</span>
        <select value={revision ? draft.id : ""} disabled={working} onChange={(e) => {
          const id = e.target.value;
          if (!id) return;
          void operation(async () => {
            const saved = await api.fetchHardwareSetup(id);
            if (!saved) throw new Error("Could not load this configuration");
            replace(saved.document, saved.revision);
            setWarnings(saved.warnings);
          });
        }}>
          <option value="">Current source / new configuration</option>
          {payload.setups.map((setup) => <option key={setup.id} value={setup.id}>{setup.label}{setup.id === payload.activeId ? " (active)" : ""}</option>)}
        </select>
      </label>
      <button disabled={working} onClick={() => {
        replace({ version: 1, id: `setup-${Date.now().toString(36)}`, label: "New hardware setup", robot: null,
          endEffector: null, teleop: null, cameras: [], mounting: {}, specializedCameras: false }, null);
        setDirty(true);
        setWarnings([]);
      }}>New setup</button>
      <button disabled={working} onClick={() => {
        replace({ ...structuredClone(draft), id: `setup-${Date.now().toString(36)}`, label: `${draft.label} copy` }, null);
        setDirty(true);
        setWarnings([]);
      }}>Duplicate setup</button>
      <button disabled={working} onClick={() => importRef.current?.click()}>Import setup</button>
      <button disabled={working || invalid} onClick={() => {
        const url = URL.createObjectURL(new Blob([JSON.stringify(draft, null, 2)], { type: "application/json" }));
        const anchor = document.createElement("a");
        anchor.href = url; anchor.download = `${draft.id}.json`; anchor.click(); URL.revokeObjectURL(url);
      }}>Export setup</button>
      <input ref={importRef} type="file" accept=".json,application/json" hidden onChange={(e) => {
        const file = e.target.files?.[0];
        if (file) void operation(async () => {
          const imported: unknown = JSON.parse(await file.text());
          if (!isHardwareDocument(imported)) throw new Error("Invalid setup document");
          replace({ ...imported, id: `setup-${Date.now().toString(36)}` }, null);
          setDirty(true);
          setWarnings([]);
        });
        e.target.value = "";
      }} />
    </div>
    <fieldset disabled={working} className="hardware-editor" key={editorKey}>
      <div className="hardware-fields">
        <label className="annotation-field"><span>Configuration name</span><input value={draft.label} maxLength={120} onChange={(e) => update({ ...draft, label: e.target.value })} /></label>
        <label className="annotation-field"><span>Configuration ID</span><input value={draft.id} readOnly={!!revision} onChange={(e) => update({ ...draft, id: e.target.value })} /></label>
      </div>
      <InstanceForm title="Robot model" kind="robot" instance={draft.robot} models={models} prefix="robot" onError={fieldError} onChange={(instance, model) => {
        if (model || !instance) {
          update(changeRobot(draft, model ?? null)); setFieldErrors({});
        } else update({ ...draft, robot: instance });
      }} />
      {draft.robot && <>
        <InstanceForm title="End effector" kind="end_effector" instance={draft.endEffector} models={toolModels} prefix="tool" onError={fieldError} onChange={(instance, model) => {
          if (model || !instance) { update(changeTool(draft, model ?? null)); setFieldErrors({}); }
          else update({ ...draft, endEffector: instance });
        }} />
        <InstanceForm title="Teleoperator" kind="teleoperator" instance={draft.teleop} models={models} prefix="teleop" onError={fieldError} onChange={(instance, model) => {
          if (model || !instance) setFieldErrors((errors) => Object.fromEntries(Object.entries(errors).filter(([key]) => !key.startsWith("teleop."))));
          if (model?.type === "spacemouse" && models.find((m) => m.kind === "end_effector" && m.type === draft.endEffector?.type)?.capabilities.includes("binary_gripper")) {
            instance = withParameter(instance!, "tool_mode", "binary");
          }
          update({ ...draft, teleop: instance });
        }} />
      </>}
      {draft.endEffector && <details className="hardware-advanced"><summary>Mounting, TCP, workspace and home</summary>
        <p className="panel-note">Standard mounting defaults follow the end-effector model. Update these values when adapters, fingers or installation geometry change.</p>
        <div className="hardware-fields">{mountingFields.map((field) => <ParameterField key={field.name} field={field} value={draft.mounting[field.name]}
          onError={(reason) => fieldError(`mounting.${field.name}`, reason)} onChange={(value) => {
            const mounting = { ...draft.mounting };
            if (value == null) delete mounting[field.name]; else mounting[field.name] = value;
            update({ ...draft, mounting });
          }} />)}</div>
      </details>}
      <div className="panel-heading"><h3>Camera instances</h3><button type="button" disabled={draft.specializedCameras} onClick={() => {
        const camera = models.find((m) => m.kind === "camera" && m.available);
        if (camera) update({ ...draft, cameras: [...draft.cameras, { name: nextCameraName(draft.cameras.map((c) => c.name)), ...instanceFor(camera) }] });
      }}>Add camera</button></div>
      <p className="panel-note">The role name is the dataset and policy camera key (for example, ee or side). Bind each role to a serial number, device path or sensor ID.</p>
      {draft.specializedCameras && <p className="panel-note">The current Thor configuration uses its specialized camera discovery pipeline. It is preserved; selecting a robot starts a configurable camera-instance list.</p>}
      {draft.cameras.map((camera, index) => <div className="hardware-camera" key={`${editorKey}:${index}`}>
        <div className="hardware-camera-heading">
          <label className="annotation-field"><span>Camera role</span><input value={camera.name} onChange={(e) => update({ ...draft, cameras: draft.cameras.map((c, i) => i === index ? { ...c, name: e.target.value } : c) })} /></label>
          <button type="button" onClick={() => { update({ ...draft, cameras: draft.cameras.filter((_, i) => i !== index) }); setFieldErrors({}); setEditorKey((key) => key + 1); }}>Remove camera</button>
        </div>
        <InstanceForm title="Camera model" kind="camera" instance={camera} models={models} prefix={`camera.${index}`} onError={fieldError} onChange={(instance, model) => {
          if (model || !instance) setFieldErrors((errors) => Object.fromEntries(Object.entries(errors).filter(([key]) => !key.startsWith(`camera.${index}.`))));
          if (instance) update({ ...draft, cameras: draft.cameras.map((c, i) => i === index ? { name: c.name, ...instance } : c) });
        }} />
      </div>)}
    </fieldset>
    {error && <p className="panel-note error" role="alert">{error}</p>}
    {message && <p className="panel-note" role="status">{message}</p>}
    {warnings.map((warning) => <p className="panel-note" key={warning}>{warning}</p>)}
    {busy && <p className="panel-note">Disconnect the running hardware session before applying another setup. Drafts can still be saved.</p>}
    <div className="control-row">
      <button disabled={working || invalid} onClick={() => void operation(async () => {
        const result = await api.validateHardware(draft);
        if (!result.ok) throw new Error(result.error ?? "Validation failed");
        setWarnings(result.warnings); setRuntimeYaml(result.runtimeYaml);
        setMessage("Parameters are valid. No hardware was connected or moved.");
      })}>Check parameters</button>
      <button disabled={working || invalid} onClick={() => void operation(() => save(false))}>Save draft</button>
      <button className="primary" disabled={working || invalid || busy} onClick={() => void operation(() => save(true))}>Save and apply</button>
      <button disabled={working || busy} onClick={() => void operation(async () => {
        const result = await api.resetHardware();
        if (!result.ok) throw new Error(result.error ?? "Reset failed");
        setPayload({ ...payload, ...result, models }); replace(result.document, result.revision);
        setMessage("Source configuration restored.");
      })}>Use source configuration</button>
    </div>
    <p className="panel-note hardware-active-label">Active: {snapshot.hardware?.label ?? "Source configuration"}. Changes apply to new sessions.</p>
    {runtimeYaml && <details className="hardware-advanced"><summary>Resolved runtime configuration</summary><pre className="hardware-runtime">{runtimeYaml}</pre></details>}
  </section>;
}
