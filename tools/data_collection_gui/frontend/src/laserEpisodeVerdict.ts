import type { LaserEpisodeVerdict } from "./types";

export type LaserEpisodeBanner = { tone: "error" | "warning" | "ok"; text: string };

const pct = (v: number | undefined) => (typeof v === "number" && v >= 0 ? `${Math.round(v * 100)}%` : "?");

/**
 * What the operator is told about the last episode's laser data.
 *
 * A discarded take is an error the operator acts on now (fix the cause, record it again);
 * an unread window or a dwell is a warning; a good one is a single quiet line.
 */
export function laserEpisodeBanner(verdict: LaserEpisodeVerdict | undefined): LaserEpisodeBanner | null {
  if (!verdict || typeof verdict.episode !== "number") return null;
  const ep = `Episode ${verdict.episode}`;
  if (verdict.rejected) {
    return {
      tone: "error",
      text:
        `${ep} 已删除：激光数据不完整，只有 ${pct(verdict.usable)} 的时长有有效激光点` +
        `（流覆盖 ${pct(verdict.coverage)}，最长断流 ${((verdict.max_gap_ms ?? 0) / 1000).toFixed(1)} s，` +
        `在 home 窝 ${pct(verdict.nest)}）。${verdict.error ?? ""} 处理后直接录下一段即可，连接不受影响。`
    };
  }
  if (verdict.kept === false) {
    // Already gone by the operator's own Discard; the laser note only explains the take.
    return {
      tone: "warning",
      text: `${ep} 已丢弃。` + (verdict.error ? `这段的激光：${verdict.error}` : "") + " 可以继续录下一段。"
    };
  }
  if (verdict.error) {
    return { tone: "warning", text: `${ep} 已保存，但激光：${verdict.error}` };
  }
  return { tone: "ok", text: `${ep} 激光数据完整：有效 ${pct(verdict.usable)}，流覆盖 ${pct(verdict.coverage)}` };
}
