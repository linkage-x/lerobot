// Which BOX is which on the BOX tab, and how fresh an arm-held BOX's 6D zero is.
//
// Every BOX reports the same factory serial; its device_id is what tells them apart, and
// tools/thor/box_devices.json says what each one is (handheld left/right, or fr3_gripper), while
// tools/fr3/fr3_arms.json says which arm carries which gripper BOX (corenetic_device_id). Both come
// with the payload-calibration snapshot (`boxes`, `arms[].gripperBox`).
//
// The two kinds are zeroed differently. A handheld BOX goes onto the vendor's fixture by hand; an
// arm BOX's fixture is the arm itself, fingertips up, reached by moving the arm
// (tools/data_collection_gui/fr3_six_d_zero.py). Zeroing an arm BOX from the generic card zeros it in
// whatever attitude the arm happens to be in, and nothing reports that afterwards -- so the tab shows
// the arm route for it, and its validity comes from that route's reports on Thor
// (outputs/fr3_six_d_zero), not from this browser's history.
//
// Pure functions only (vitest: boxRoles.test.ts).

import type { DeviceStatus } from "../types";
import type { CalibrationKind } from "./config";
import type { PayloadCalibrationResponse, ZeroReport } from "./payloadCalibration";

export type BoxRole = {
  kind: "handheld" | "fr3_gripper" | "unregistered";
  /** device_id as a string; "" when no row of the group carries one */
  deviceId: string;
  /** handheld: the operator's hand */
  hand?: "left" | "right";
  /** fr3_gripper: the arm whose fr3_arms.json entry names this BOX ("" when no arm does) */
  arm?: string;
  /** heading, e.g. 手持右 BOX 3906437858 / 右臂夹爪 BOX 3100715645 */
  title: string;
  /** shown under the heading when something is missing from the registries */
  note?: string;
};

const SIDE: Record<string, string> = { left: "左", right: "右" };

/** The BOX's device_id: the `box<id>` namespace of a multi-BOX rig, else a row's config.device_id. */
export function boxDeviceNumber(boxId: string, devices: DeviceStatus[]): string {
  const m = /^box(\d+)$/.exec(boxId);
  if (m) return m[1];
  for (const d of devices) {
    const raw = d.config?.device_id;
    if (raw != null && String(raw).trim()) return String(raw).trim();
  }
  return "";
}

/** What this BOX is, from box_devices.json and fr3_arms.json; ``payload`` null = not loaded (yet). */
export function resolveBoxRole(deviceId: string, payload: PayloadCalibrationResponse | null): BoxRole {
  const name = deviceId ? `BOX ${deviceId}` : "BOX";
  if (!payload) return { kind: "unregistered", deviceId, title: name };
  const armRow = deviceId ? payload.arms.find((a) => a.gripperBox != null && String(a.gripperBox) === deviceId) : undefined;
  const entry = deviceId ? payload.boxes.find((b) => b.deviceId === deviceId) : undefined;
  if (armRow || entry?.role === "fr3_gripper") {
    const arm = armRow?.arm ?? "";
    const side = SIDE[arm] ?? arm;
    return {
      kind: "fr3_gripper",
      deviceId,
      arm,
      title: arm ? `${side}臂夹爪 ${name}` : `FR3 夹爪 ${name}`,
      note: !arm
        ? "box_devices.json 登记为 FR3 夹爪，但 tools/fr3/fr3_arms.json 里没有哪条臂的 corenetic_device_id 是它：下面的零点要手动选臂，先找维护者补上对应关系。"
        : !entry
          ? "fr3_arms.json 里这条臂的夹爪是它，但 box_devices.json 还没登记：6D 拟合登记时会补上。"
          : undefined,
    };
  }
  if (entry?.role === "handheld") {
    const hand = entry.hand === "left" || entry.hand === "right" ? entry.hand : undefined;
    return { kind: "handheld", deviceId, hand, title: `手持${hand ? SIDE[hand] : ""} ${name}` };
  }
  return {
    kind: "unregistered",
    deviceId,
    title: name,
    note: deviceId
      ? `这只 BOX 没有在 tools/thor/box_devices.json 里登记角色，下面按手持 BOX 显示。装在机械臂上的 BOX 不要用这里的手动校零：先找维护者登记（role: fr3_gripper，并写进 fr3_arms.json）。`
      : "认不出这只 BOX 的 device_id，下面按手持 BOX 显示。",
  };
}

/** Handheld first (the usual daily work), then unregistered, then the arm BOXes; stable within a kind. */
export function boxRoleRank(role: BoxRole): number {
  return role.kind === "handheld" ? 0 : role.kind === "unregistered" ? 1 : 2;
}

/** ``2026-11-10T15:04:05`` (the report's local time, as fr3_six_d_zero writes it) -> ms epoch. */
function reportTime(created: string): number | null {
  if (!created) return null;
  const t = new Date(created).getTime();
  return Number.isFinite(t) ? t : null;
}

/**
 * When the arm route last zeroed this BOX successfully, per kind (ms epoch, null = never).
 *
 * Only passing reports for this arm and this BOX count. An origin run ends with a dynamic zero too,
 * so it refreshes both; a dynamic run refreshes only the dynamic zero.
 */
export function armZeroTimes(
  reports: ZeroReport[] | undefined,
  arm: string,
  deviceId: string,
): Partial<Record<CalibrationKind, number | null>> {
  let origin: number | null = null;
  let dynamic: number | null = null;
  for (const r of reports ?? []) {
    if (!r.pass || (arm && r.arm !== arm) || String(r.deviceId ?? "") !== deviceId) continue;
    const t = reportTime(r.created);
    if (t == null) continue;
    if (dynamic == null || t > dynamic) dynamic = t;
    if (r.mode === "origin" && (origin == null || t > origin)) origin = t;
  }
  return { force_origin: origin, force_dynamic: dynamic };
}
