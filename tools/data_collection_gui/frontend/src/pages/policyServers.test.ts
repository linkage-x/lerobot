import { describe, expect, it } from "vitest";
import type { PolicyServerEntry, PolicyServerListing } from "../types";
import { LOCAL_POLICY, defaultPolicyChoice, describeServer, holdsCheckpoint, startNote } from "./policyServers";

const CKPT = "outputs/train/fr3_insert__pi05/checkpoints/030000/pretrained_model";

function server(overrides: Partial<PolicyServerEntry> = {}): PolicyServerEntry {
  return {
    address: "192.168.1.179:8790",
    reachable: true,
    status: {
      loaded: { spec: { checkpoint: CKPT }, fingerprint: {}, device: "cuda" },
      loading: null,
      holder: null,
      idle_s: 12,
      idle_unload_s: 900,
      gpu: { name: "RTX 4090 D", free_mb: 14 * 1024, total_mb: 24 * 1024, allocated_mb: 9 * 1024 }
    },
    ...overrides
  };
}

function listing(localAvailable: boolean, servers: PolicyServerEntry[]): PolicyServerListing {
  return { local: { available: localAvailable, detail: "" }, servers };
}

describe("defaultPolicyChoice", () => {
  it("opens a GPU-less robot host on its policy server", () => {
    expect(defaultPolicyChoice(listing(false, [server()]))).toBe("192.168.1.179:8790");
  });

  it("keeps this machine where it has a GPU", () => {
    expect(defaultPolicyChoice(listing(true, [server()]))).toBe(LOCAL_POLICY);
  });

  it("restores the last choice only while it is still usable", () => {
    expect(defaultPolicyChoice(listing(true, [server()]), "192.168.1.179:8790")).toBe("192.168.1.179:8790");
    expect(defaultPolicyChoice(listing(true, [server({ reachable: false })]), "192.168.1.179:8790")).toBe(LOCAL_POLICY);
    expect(defaultPolicyChoice(listing(false, [server()]), LOCAL_POLICY)).toBe("192.168.1.179:8790");
  });
});

describe("server notes", () => {
  it("recognises the loaded checkpoint by its id", () => {
    expect(holdsCheckpoint(server(), "fr3_insert__pi05/030000")).toBe(true);
    expect(holdsCheckpoint(server(), "fr3_insert__pi05/020000")).toBe(false);
  });

  it("warns before a start that has to load, connect to a dead host, or share the GPU", () => {
    expect(startNote(server(), "fr3_insert__pi05/030000")).toBe("");
    expect(startNote(server(), "fr3_insert__pi05/020000")).toMatch(/loads this checkpoint/);
    expect(startNote(server({ reachable: false, error: "refused" }), "x/1")).toMatch(/not answering/);
    const busy = server();
    busy.status = { ...busy.status!, holder: "192.168.1.156:40000" };
    expect(startNote(busy, "fr3_insert__pi05/030000")).toMatch(/serving another rollout/);
  });

  it("describes what the server holds and how full its GPU is", () => {
    expect(describeServer(server())).toBe("holds fr3_insert__pi05/030000 · RTX 4090 D 10.0/24.0 GB in use");
    expect(describeServer(server({ reachable: false, error: "ConnectionRefusedError" }))).toMatch(/^unreachable/);
  });
});
