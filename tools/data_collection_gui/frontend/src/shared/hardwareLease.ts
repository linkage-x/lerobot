// The banner for hardware time-sharing between production and the maintainer's dev gateway
// (gateway: tools/data_collection_gui/hardware_lease.py; deploy: run/deploy.sh <thor> --dev).
import type { HardwareLease } from "../types";

export type LeaseBanner = {
  tone: "info" | "warn" | "alert";
  text: string;
  action: "borrow" | "return" | "reclaim" | null;
  label: string;
  /** What the confirm dialog asks before the action. */
  confirm: string;
};

/** run/deploy.sh --dev refreshes these from production on every deploy (run/README.md section 7). */
export const DEV_CALIBRATION_NOTE =
  " 注意：每次部署开发网关都会用生产的版本覆盖 outputs/calibration（相机标定、同球检验、BOX 6D 采集）和" +
  " world/门禁文件，在这里做的标定结果会被删掉，也不会进入生产；要保留就在重新部署前拷出来，或者回生产网关上做标定。";

const since = (lease: HardwareLease) => (lease.since ? lease.since.replace("T", " ").replace("Z", " UTC") : "?");

/** What to show for the lease, or null when nothing needs saying (production holding its own hardware). */
export function leaseBanner(lease: HardwareLease | null | undefined, commit = "", branch = ""): LeaseBanner | null {
  if (!lease) return null;
  if (lease.holder === "unknown") {
    return {
      tone: "alert",
      text: "硬件租约文件无法读取：生产和开发网关都不会使用相机、BOX、激光跟踪仪和机械臂。",
      action: lease.role === "production" ? "reclaim" : null,
      label: "收回硬件",
      confirm: "把硬件收回给生产网关？"
    };
  }
  if (lease.role === "production") {
    if (lease.mine) return null;
    return {
      tone: "warn",
      text:
        `硬件已借给开发网关（${lease.devUrl || "dev"}，${since(lease)} 起${lease.by ? `，${lease.by}` : ""}）：` +
        "这里暂时不能用相机、BOX、激光跟踪仪和机械臂，离线功能照常。",
      action: "reclaim",
      label: "收回硬件",
      confirm:
        "收回硬件：先请开发网关归还；它正在用硬件时会拒绝，并说明原因。\n" +
        "如果开发网关已经没有响应，会结束它目录里的全部进程（录制器、rollout、replay 等，包括正在动的机械臂）后再收回。\n继续？"
    };
  }
  const version = [commit ? commit.slice(0, 8) : "", branch ? `(${branch})` : ""].filter(Boolean).join(" ");
  if (lease.mine) {
    return {
      tone: "alert",
      text:
        `开发网关${version ? ` ${version}` : ""}正持有硬件（${since(lease)} 起）：生产网关的硬件功能已暂停，` +
        "学生不能录制。用完请归还。" + DEV_CALIBRATION_NOTE,
      action: "return",
      label: "归还硬件",
      confirm: "把硬件还给生产网关？录制器、replay、rollout 等必须已经停下。"
    };
  }
  return {
    tone: "info",
    text:
      `开发网关${version ? ` ${version}` : ""}：没有借用硬件，相机、BOX、激光跟踪仪和机械臂不可用。` +
      "借用时生产网关必须空闲（录制器未连接，没有 replay、rollout 或标定在运行）。" + DEV_CALIBRATION_NOTE,
    action: "borrow",
    label: "借用硬件",
    confirm:
      "借用硬件：借出期间生产网关停止所有硬件功能（学生不能录制），直到你在这里归还。\n" +
      "机械臂运动仍然只能由现场的人启动，并且手放在急停上。\n继续？"
  };
}
