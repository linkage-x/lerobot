// The calibration page's tabs: what each holds and when an operator needs it. Order = how often.

export type CalibrationTabId = "overview" | "cameras" | "handheld" | "fr3" | "box" | "laser" | "log";

export type CalibrationTab = {
  id: CalibrationTabId;
  label: string;
  hint: string;
  when: string;
  contents: string;
};

export const CALIBRATION_TABS: CalibrationTab[] = [
  {
    id: "overview",
    label: "概览",
    hint: "每天开工先看：有没有要重做的",
    when: "每天开工、相机被碰过之后",
    contents: "相机自检 + 本页导航",
  },
  {
    id: "box",
    label: "BOX 传感器",
    hint: "6D 零点/origin（手持上工装，臂上由机械臂摆姿态）、触觉校准、6D 负载拟合",
    when: "每次上电（零点）、换末端（6D 拟合）、按有效期",
    contents: "按 BOX 分组（手持在前、臂上单列）：预热计时、origin、动态校准、触觉校准、6D 负载拟合 K",
  },
  {
    id: "handheld",
    label: "手持末端",
    hint: "marker→TCP（pivot）、手眼旋转",
    when: "新 rig / 贴纸重贴 / 换指尖后",
    contents: "marker→TCP pivot 解算、手眼旋转",
  },
  {
    id: "fr3",
    label: "FR3 机械臂",
    hint: "末端负载辨识、基座注册边、球窝 pivot",
    when: "末端装拆东西后（负载）、机械臂挪动后（注册边）",
    contents: "末端负载辨识（Desk）、world→fr3_base 注册边、同球检验",
  },
  {
    id: "cameras",
    label: "相机",
    hint: "多相机标定、内参覆盖、世界系、跨相机一致性",
    when: "自检报漂移、支架改动、换镜头",
    contents: "标定向导、内参覆盖、世界系检查、跨相机一致性",
  },
  {
    id: "laser",
    label: "激光跟踪仪",
    hint: "跟踪仪安装常量",
    when: "跟踪仪挪位或靶座重装后",
    contents: "跟踪仪站位 T_WG 与靶座偏置",
  },
  {
    id: "log",
    label: "日志与历史",
    hint: "BOX 校准日志和本机历史",
    when: "排查问题时",
    contents: "操作日志、校准历史",
  },
];

const TAB_KEY = "lerobot.calibration.tab";

export function isTabId(value: unknown): value is CalibrationTabId {
  return CALIBRATION_TABS.some((t) => t.id === value);
}

export function readStoredTab(): CalibrationTabId {
  // ?caliTab=fr3 links straight to a tab; otherwise the one last used here
  const linked = new URLSearchParams(window.location.search).get("caliTab");
  if (isTabId(linked)) return linked;
  try {
    const v = window.localStorage.getItem(TAB_KEY);
    return isTabId(v) ? v : "overview";
  } catch {
    return "overview";
  }
}

export function storeTab(tab: CalibrationTabId): void {
  try {
    window.localStorage.setItem(TAB_KEY, tab);
  } catch {
    /* ignore storage denial */
  }
}
