import { describe, expect, it } from "vitest";
import { laserEpisodeBanner } from "./laserEpisodeVerdict";

describe("laserEpisodeBanner", () => {
  it("says nothing before an episode has been stopped", () => {
    expect(laserEpisodeBanner(undefined)).toBeNull();
    expect(laserEpisodeBanner({})).toBeNull();
  });

  it("names a discarded take, the numbers, and the cause", () => {
    const banner = laserEpisodeBanner({
      episode: 0, rejected: true, incomplete: true, usable: 0.132, coverage: 0.132, max_gap_ms: 4442, nest: 0,
      error: "is RadianCAL-Pro or another tracker client running on the capture PC?"
    });
    expect(banner?.tone).toBe("error");
    expect(banner?.text).toContain("Episode 0 已删除");
    expect(banner?.text).toContain("13%");
    expect(banner?.text).toContain("4.4 s");
    expect(banner?.text).toContain("RadianCAL");
  });

  it("says an operator-discarded take is gone, not that its data was kept", () => {
    const banner = laserEpisodeBanner({ episode: 0, rejected: false, kept: false, error: "beam off the SMR" });
    expect(banner?.tone).toBe("warning");
    expect(banner?.text).toContain("Episode 0 已丢弃");
    expect(banner?.text).toContain("继续录下一段");
  });

  it("keeps a kept episode's problem as a warning and a good one quiet", () => {
    const kept = laserEpisodeBanner({ episode: 2, kept: true, error: "could not read" });
    expect(kept?.tone).toBe("warning");
    expect(kept?.text).toContain("已保存");
    const ok = laserEpisodeBanner({ episode: 3, usable: 0.97, coverage: 1, error: "" });
    expect(ok?.tone).toBe("ok");
    expect(ok?.text).toContain("97%");
  });
});
