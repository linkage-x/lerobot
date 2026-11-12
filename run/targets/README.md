# Deployment targets and stations

One JSON file per machine that `run/` scripts reach. Nothing under `run/` may hard-code an address
(`tests/scripts/test_run_targets.py` checks it).

```bash
python3 run/targets.py list                  # every target, how you reach it, its station
bash run/deploy.sh thor2                     # production on thor2 (same guard as everywhere)
bash run/deploy.sh thor1 --sandbox zhangsan  # a sandbox on thor1
bash run/sandbox.sh thor2 list
bash run/kill_thor_gateway.sh thor2
```

**There is no default target.** Every script that acts on a machine needs it named, and without one
it stops and lists the targets: with two Thors, a forgotten name must not quietly restart thor1's
gateway (and kill its recorder) when thor2 was meant. `thor` stays an alias of `thor1`, so
`bash run/deploy.sh thor --sandbox NAME` keeps working; the hostname check does not help here, since
thor1 *is* a valid target.

## Two concepts

- **Target** = a machine you deploy to (a Thor, the FR3 workstation). File: `run/targets/<name>.json`.
- **Station** = one recording rig: cameras, BOXes, arms, a world frame and its calibration. A target
  names the station it belongs to (`"station"`, `null` for a machine that is not part of one).
  Several machines may belong to one station; a station's calibration only ever describes its own
  cameras.

## A target file

| key | meaning |
|---|---|
| `station` | the station this machine belongs to, or `null` |
| `user`, `address` | how it is reached *on its own LAN* (`user@address`, gateway `http://address:port`) |
| `hostname` | the machine's real hostname; checked over ssh before every deploy, sandbox command and gateway kill |
| `repo_dir` | production tree on the machine |
| `profile`, `config` | what the gateway runs as (`thor` / `workstation`) and with which config |
| `gateway_port` | production gateway port |
| `sandboxes` | whether students get sandboxes there (Thor profile only) |
| `calibration_in_tree` | `true` only for the machine whose cameras this tree's committed calibration describes (see "Not done yet") |
| `box_net_addr` | the static address the legacy BOX firmware pushes UDP to, or `null`. Two Thors on one LAN must never both claim one |
| `aliases`, `note` | optional |

**Hostnames must be unique.** A cloned image keeps the old hostname (and `machine-id`, so DHCP hands
out the same lease too); set a new hostname on the clone and run `systemd-machine-id-setup` after
removing `/etc/machine-id`. The registry refuses two targets with one hostname, and the identity
check refuses a machine whose hostname is not the one in its file. That is what stops a deploy when
a DHCP address moves to another machine.

## Reaching a machine on another LAN

The committed file says what the machine *is*. How *you* reach it is personal and goes in the
git-ignored `run/targets.local.json`. Only `ssh` and `gateway_url` can be overridden:

```json
{ "thor2": { "ssh": "thor2-lab", "gateway_url": "http://127.0.0.1:18765" } }
```

```sshconfig
# ~/.ssh/config: everything (ssh, rsync, the deploy scripts) goes through the jump host
Host thor2-lab
  HostName 192.168.1.152
  User nvidia
  ProxyJump gateway.example-lab
  LocalForward 18765 127.0.0.1:8765   # the GUI reaches the gateway through this tunnel
```

Because the hostname is checked whatever route is used, a wrong alias cannot deploy to the wrong
machine. A sandbox on a tunnelled target needs its own forward (deploy prints a warning).

## Not done yet (in the order they will matter)

1. **Per-station files.** The tree still holds one station's calibration and hardware description:
   `tools/thor/gmsl2/world/*.json`, `camera_identity_expected.json`, `camera_intrinsics_registry.json`,
   `box_time_delays.json`, `cart_tags.json`, `tools/thor/box_devices.json`, `tools/fr3/fr3_arms.json`,
   the camera config yaml and the tracking config's calibration pointers. Until those live under
   `stations/<station>/` and deploy picks the target's station, only the target with
   `calibration_in_tree: true` gets the EE calibration inputs; sync warns on any other Thor, and
   Connect refuses to record there because the camera serials do not match. Moving these files is
   calibration data (AGENTS.md section 4): a maintainer decides the layout.
2. **Device allowlist per station.** Each station connects only to its own BOX ids, arms and laser
   tracker, never to "whatever discovery finds". Needed even with separate VLANs (a BOX plugged into
   the wrong switch).
3. **Station in the data.** Episode meta and label provenance must record the station (and the
   machine's hostname), and export must refuse mixing stations, like it refuses mixing extrinsics.
   Today DEPLOYED.json and the GUI title show the target and station; datasets do not yet.
4. **Machine bootstrap.** `tools/thor/DEPLOYMENT.md` as a re-runnable script: apt, Python
   environments (`.venv`, `third_party/opencv_kalibr/.venv`, `~/Code/infer/.venv-fr3`), BOX SDK,
   camera overlay/ISP files (the kernel modules must match the L4T version), network, headless.
5. **Tunnel automation.** If many people work from outside the lab, deploy could open the forward
   itself instead of relying on `~/.ssh/config`.
