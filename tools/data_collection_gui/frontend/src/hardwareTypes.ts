export type HardwareKind = "robot" | "end_effector" | "camera" | "teleoperator";

export type HardwareField = {
  name: string;
  label: string;
  kind: "string" | "device" | "integer" | "number" | "boolean" | "select" | "json";
  required: boolean;
  nullable: boolean;
  advanced: boolean;
  default: unknown;
  options?: (string | number)[] | null;
};

export type HardwareModel = {
  kind: HardwareKind;
  type: string;
  label: string;
  fields: HardwareField[];
  compatibleRobots: string[];
  mounting: Record<string, unknown>;
  capabilities: string[];
  description: string;
  available: boolean;
  error: string;
};

export type HardwareInstance = { type: string; params: Record<string, unknown> };
export type HardwareCamera = HardwareInstance & { name: string };
export type HardwareDocument = {
  version: 1;
  id: string;
  label: string;
  robot: HardwareInstance | null;
  endEffector: HardwareInstance | null;
  teleop: HardwareInstance | null;
  cameras: HardwareCamera[];
  mounting: Record<string, unknown>;
  specializedCameras: boolean;
};

export type HardwareSetupEntry = { id: string; label: string; revision: string };
export type HardwareSavedSetup = {
  document: HardwareDocument;
  revision: string;
  runtime: Record<string, unknown>;
  warnings: string[];
};
export type HardwarePayload = {
  document: HardwareDocument;
  revision: string | null;
  activeId: string | null;
  setups: HardwareSetupEntry[];
  models: HardwareModel[];
  sourcePath: string;
  runtimePath: string;
  busyReason: string;
  warnings: string[];
  capabilities: string[];
};

export type HardwareResult<T> = T & { ok: boolean; error?: string };
