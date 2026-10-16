// Wording for the FR3 gripper teleop (tools/fr3/fr3_gripper_teleop.py), kept out of
// the component so a test can pin it.
//
// The FR3's gripper is a BOX like the handheld one: its jaw follows the handheld's
// jaw, or shuts once the handheld trigger is pressed home (the replay's rule). It
// starts following only once the two agree (soft takeover), and exit leaves the
// arm's BOX in control mode at the last target -- a grip keeps its force.
import type { GripperTeleopJob, GripperTeleopResponse } from "../types";

/** Commanded shut but stopped this far open: something is in the jaws. */
const GRIPPING_MIN_WIDTH_MM = 3;

export function isTeleopRunning(job: GripperTeleopJob | null | undefined): boolean {
  return job?.state === "running";
}

function mm(v: number | null | undefined): string {
  return v == null ? "—" : `${v.toFixed(1)} mm`;
}

/** "Gripping" when the arm was told to close past where it stopped. */
export function grippingNote(targetMm: number | null | undefined, armWidthMm: number | null | undefined): string {
  if (targetMm == null || armWidthMm == null) return "";
  return armWidthMm - targetMm >= GRIPPING_MIN_WIDTH_MM ? `（夹着东西：目标 ${mm(targetMm)}，停在 ${mm(armWidthMm)}）` : "";
}

export function teleopStatusText(payload: GripperTeleopResponse | null): { text: string; warn: boolean } {
  const job = payload?.job;
  const s = job?.status;
  switch (job?.state) {
    case "running":
      if (!s) return { text: "启动中：连接 BOX…", warn: false };
      if (s.state === "waiting") {
        return {
          text:
            `等待对齐：手持指令 ${mm(s.commandMm)}，FR3 保持 ${mm(s.targetMm)}。` +
            "把手持夹爪调到与 FR3 相差 5 mm 以内才开始跟随（防止一进来就把夹着的东西松开）",
          warn: false,
        };
      }
      if (!s.handheldFresh) return { text: "手持 BOX 没有新数据：FR3 夹爪保持不动", warn: true };
      return {
        text:
          `跟随中：手持 ${mm(s.handheldWidthMm)} · 扳机 ${s.triggerPct == null ? "—" : `${s.triggerPct.toFixed(0)}%`}` +
          ` → FR3 目标 ${mm(s.targetMm)}（实测 ${mm(s.armWidthMm)}）${grippingNote(s.targetMm, s.armWidthMm)}`,
        warn: false,
      };
    case "exited": {
      const kept = s?.armMode === 1;
      return {
        text: kept
          ? `已退出遥控：FR3 夹爪保持 ${mm(s?.targetMm)}，仍在控制模式，不卸力${grippingNote(s?.targetMm, s?.armWidthMm)}`
          : `已退出遥控，但 FR3 夹爪 BOX 模式读回 ${s?.armMode ?? "失败"}：可能没保持住，看一眼夹爪`,
        warn: !kept,
      };
    }
    case "error":
      return { text: `失败${job.returncode != null ? `（exit ${job.returncode}）` : ""}：${job.error || "见日志"}`, warn: true };
    default: {
      const hold = payload?.hold;
      if (hold?.target_m != null && hold.mode === 1) {
        return { text: `未在遥控。上次退出时 FR3 夹爪保持 ${mm(hold.target_m * 1e3)}（${hold.written}）`, warn: false };
      }
      return { text: "未在遥控", warn: false };
    }
  }
}

/** Warnings that outlive one status line. */
export function teleopWarnings(payload: GripperTeleopResponse | null): string[] {
  const out: string[] = [];
  if (payload && payload.armDeviceId == null) out.push("回放配置里没有 robot.corenetic_device_id，不知道哪台 BOX 是 FR3 的夹爪");
  const reboots = payload?.job?.status?.armReboots ?? 0;
  if (reboots > 0) out.push(`FR3 夹爪 BOX 重启过 ${reboots} 次（已重新接管）：查手腕上 BOX 的供电/线缆`);
  return out;
}
