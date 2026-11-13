import type { HardwareDocument, HardwareField, HardwareInstance, HardwareModel } from "../hardwareTypes";

export function instanceFor(model: HardwareModel): HardwareInstance {
  return {
    type: model.type,
    params: Object.fromEntries(model.fields.filter((f) => f.default != null).map((f) => [f.name, structuredClone(f.default)]))
  };
}

export function changeRobot(document: HardwareDocument, model: HardwareModel | null): HardwareDocument {
  return {
    ...document,
    robot: model ? instanceFor(model) : null,
    endEffector: null,
    mounting: {},
    specializedCameras: model ? false : document.specializedCameras,
    // A leader for another embodiment must be deliberately reselected.
    teleop: null
  };
}

export function changeTool(document: HardwareDocument, model: HardwareModel | null): HardwareDocument {
  return {
    ...document,
    endEffector: model ? instanceFor(model) : null,
    teleop: model?.capabilities.includes("binary_gripper") && document.teleop?.type === "spacemouse"
      ? withParameter(document.teleop, "tool_mode", "binary") : document.teleop,
    // Do not carry a previous tool's TCP or home over to a different installation.
    mounting: model ? structuredClone(model.mounting) : {}
  };
}

export function parseField(field: HardwareField, text: string): unknown {
  if (text === "") {
    if (field.required) throw new Error("Required");
    return field.nullable ? null : undefined;
  }
  if (field.kind === "json") return JSON.parse(text);
  if (field.kind === "number" || field.kind === "integer") {
    const value = Number(text);
    if (!Number.isFinite(value) || (field.kind === "integer" && !Number.isInteger(value))) {
      throw new Error(field.kind === "integer" ? "Enter an integer" : "Enter a finite number");
    }
    return value;
  }
  if (field.kind === "device" && /^\d+$/.test(text)) return Number(text);
  return text;
}

export function withParameter(instance: HardwareInstance, name: string, value: unknown): HardwareInstance {
  const params = { ...instance.params };
  if (value === undefined) delete params[name];
  else params[name] = value;
  return { ...instance, params };
}

export function nextCameraName(names: string[]): string {
  let number = 1;
  while (names.includes(`camera_${number}`)) number += 1;
  return `camera_${number}`;
}

export function isHardwareDocument(value: unknown): value is HardwareDocument {
  if (typeof value !== "object" || value === null) return false;
  const doc = value as Partial<HardwareDocument>;
  const instance = (item: unknown): boolean => {
    if (item === null) return true;
    if (typeof item !== "object" || item === null) return false;
    const candidate = item as Partial<HardwareInstance>;
    return typeof candidate.type === "string" && typeof candidate.params === "object" && candidate.params !== null && !Array.isArray(candidate.params);
  };
  return doc.version === 1 && typeof doc.id === "string" && typeof doc.label === "string"
    && instance(doc.robot) && instance(doc.endEffector) && instance(doc.teleop)
    && typeof doc.mounting === "object" && doc.mounting !== null && !Array.isArray(doc.mounting)
    && typeof doc.specializedCameras === "boolean" && Array.isArray(doc.cameras)
    && doc.cameras.every((camera) => camera !== null && instance(camera) && typeof camera.name === "string");
}
