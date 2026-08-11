// What the operator actually has to do, per step.
//
// This text is the feature. The 0804 calibration failed its first round not
// because the maths was wrong but because nobody knew how far to push the board:
// coverage came out at 48-71% of frame radius, the distortion fits folded over
// inside their own frames, and the whole capture had to be redone. The recapture
// that fixed it followed exactly the protocol below and took cam_08 from 62% to
// 96%. Putting it in a document instead of in the panel is how that gets lost
// again.
import type { CalibrationStepKind } from "../types";

export type GuideStep = { title: string; detail?: string };

export const BOARD_NOTE =
  "标定板用 A 板（charuco_400，12×9 格，30 mm 方格）。B 板（charuco_300，15 mm marker）在 1.2 m 处实测检出 0/63，只能在 0.7 m 以内用。" +
  "两块板的 marker ID 重叠（A 是 0–53，B 是 0–62），绝对不能同时出现在画面里。";

export const INTRINSICS_GUIDE: GuideStep[] = [
  {
    title: "只面对当前这一台相机，其余相机此刻不用管",
    detail: "内参是逐台标的：约束来自这一台画幅被覆盖了多少，别的相机看不看得到板都不影响它。",
  },
  {
    title: "站到相机侧后方，手持标定板，距离 0.5–0.8 m",
    detail: "这个距离上 A 板的 marker 成像 25–45 px，检测最稳。太远会漏检，太近覆盖不到边角。",
  },
  {
    title: "把板贴到画面的四个角和四条边——四个角必须都走到",
    detail: "这是整个流程里最容易做不到位、也最致命的一步。画幅外侧没有数据时，畸变模型在那里是外推，可能在画幅内就折返失效。",
  },
  {
    title: "每个边缘位置做 ±30–45° 倾斜",
    detail: "正对着板拍无法区分焦距与畸变，倾斜才能把它们解开。",
  },
  {
    title: "边录边看预览：任何时刻都应有一部分板贴着画面边框",
    detail: "这是现场唯一可用的合格判据。如果整段录制里板始终在画面中间，这一台就要重录。",
  },
  { title: "录 30–60 秒后点「保存本段」", detail: "参考：cam_08 那次 605 帧里 500 帧有效，是迄今最好的一次。" },
];

export const EXTRINSICS_GUIDE: GuideStep[] = [
  {
    title: "把板举到多台相机能同时看见的区域（桌面中部上方）",
    detail: "外参靠「同一时刻多台相机看到同一块板」建立约束。只有一台看得到的帧，对外参没有任何贡献。",
  },
  {
    title: "缓慢挥动，走遍整个作业区，让不同的相机组合轮流同时看到板",
    detail: "相机之间要通过共视连成一张图；某两台从不同时看到板，它们的相对位姿就只能靠链式传递，误差会累积。",
  },
  { title: "板面尽量朝向相机群，不要长时间只对着某一台" },
  { title: "动作要慢——运动模糊会让角点定位变差", detail: "相机之间是硬同步的（SOF 偏差 0.008–0.009 ms），所以慢挥不会引入跨相机的时间误差，只影响清晰度。" },
  { title: "录 60 秒左右后点「保存本段」" },
];

export function guideFor(kind: CalibrationStepKind): GuideStep[] {
  return kind === "intrinsics" ? INTRINSICS_GUIDE : EXTRINSICS_GUIDE;
}

export function stepTitle(kind: CalibrationStepKind, camera: string): string {
  return kind === "intrinsics" ? `内参采集 · ${camera}` : "外参采集 · 所有相机协同";
}

export const STEP_STATUS_LABEL: Record<string, string> = {
  pending: "待录制",
  recording: "录制中",
  captured: "已完成",
  skipped: "已跳过",
};

export const STEP_STATUS_DOT: Record<string, string> = {
  pending: "idle",
  recording: "warning",
  captured: "running",
  skipped: "idle",
};

/** Skipping a camera is normal; skipping the shared capture leaves nothing to solve. */
export function skipConsequence(kind: CalibrationStepKind): string {
  return kind === "intrinsics"
    ? "跳过后这一台沿用现有内参。视野被遮挡的相机（如 cam_00）应当跳过。"
    : "外参采集不能跳过——没有它就没有相机之间的位姿，无法解算。";
}
