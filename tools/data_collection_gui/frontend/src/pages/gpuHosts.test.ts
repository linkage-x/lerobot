import { describe, expect, it } from "vitest";
import type { GpuHostStatus, GpuService } from "../types";
import { buttonsFor, describeService, duration, opNote } from "./gpuHosts";

const policy = (over: Partial<GpuService> = {}): GpuService => ({
  id: "policy_server",
  label: "FR3 policy server (:8790)",
  port: 8790,
  gpuGb: 11,
  state: "running",
  pids: [1],
  upS: 3600,
  gpuMb: 11700,
  loaded: "job/045000",
  loading: false,
  holder: null,
  ...over
});
const wam = (over: Partial<GpuService> = {}): GpuService => ({
  id: "wam_jepa",
  label: "WAM",
  port: 8791,
  gpuGb: 12,
  state: "stopped",
  pids: [],
  upS: null,
  gpuMb: 0,
  units: [],
  ...over
});
const host = (services: GpuService[], over: Partial<GpuHostStatus> = {}): GpuHostStatus => ({
  host: "Dante",
  gpu: { name: "RTX 5090", total_mb: 32607, used_mb: services.reduce((sum, svc) => sum + svc.gpuMb, 1400) },
  services,
  others: [],
  training: [],
  op: null,
  ...over
});

describe("gpu host panel", () => {
  it("describes a policy server by what it holds and who it serves", () => {
    expect(describeService(policy())).toBe("running · 11.4 GB · up 60 min · holds job/045000");
    expect(describeService(policy({ holder: "10.77.0.156:51234" }))).toContain("serving 10.77.0.156:51234");
    expect(describeService(policy({ holder: "warmup", loading: true }))).not.toContain("serving");
    expect(describeService(wam())).toBe("stopped");
    expect(describeService(wam({ state: "starting", upS: 20 }))).toBe("starting · up 20 s · waiting for :8791");
  });

  it("offers switch when another service runs, and start only when the memory is free", () => {
    const status = host([policy(), wam()]);
    const buttons = buttonsFor(status, status.services[1], false);
    expect(buttons.map((b) => [b.action, b.disabledReason])).toEqual([
      ["switch", ""],
      ["start", ""]
    ]);
    expect(buttons[0].label).toContain("stops policy_server");
    const full = host([policy(), wam()], { gpu: { name: "x", total_mb: 32607, used_mb: 25000 } });
    expect(buttonsFor(full, full.services[1], false)[1].disabledReason).toMatch(/needs 12 GB, 7.4 GB free/);
  });

  it("refuses to stop a policy server serving a rollout, here or elsewhere", () => {
    const status = host([policy({ holder: "10.77.0.156:5" }), wam()]);
    expect(buttonsFor(status, status.services[0], false)[0].disabledReason).toBe("serving 10.77.0.156:5");
    expect(buttonsFor(status, status.services[1], false)[0].disabledReason).toMatch(/policy_server is serving/);
    const warming = host([policy({ holder: "warmup" }), wam()]);
    expect(buttonsFor(warming, warming.services[0], false)[0].disabledReason).toBe("");
  });

  it("greys everything during a rollout, an action in flight, or a training run", () => {
    const status = host([policy(), wam()]);
    expect(buttonsFor(status, status.services[0], true)[0].disabledReason).toMatch(/rollout is running/);
    const busy = host([policy(), wam()], {
      op: { action: "switch", service: "wam_jepa", state: "running", message: "", startedAt: 0, finishedAt: null }
    });
    expect(buttonsFor(busy, busy.services[0], false)[0].disabledReason).toBe("busy: switch wam_jepa");
    const training = host([wam()], { training: ["pid 7 (lerobot_train.py)"] });
    expect(buttonsFor(training, training.services[0], false)[0].disabledReason).toMatch(/training on this card/);
  });

  it("reports the last action", () => {
    expect(opNote(null)).toBe("");
    expect(
      opNote({ action: "stop", service: "wam_jepa", state: "failed", message: "boom", startedAt: 100, finishedAt: 110 }, 130)
    ).toBe("stop wam_jepa FAILED 20 s ago: boom");
    expect(duration(3 * 3600 + 120)).toBe("3 h 2 min");
  });
});
