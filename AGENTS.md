# AGENTS.md — rules for anyone (human or AI agent) changing this repository

This file is for every coding agent (Codex, Claude Code, Cursor, …) and for the people driving them.
`CLAUDE.md` only imports this file, so there is one set of rules. Read all of it before changing code.

This is a fork of LeRobot used to run a real lab: a Thor machine with 11 GMSL2 cameras records handheld
grippers (BOX sensors), the GUI generates end-effector trajectories from AprilTags, and two Franka FR3 arms
replay them. Mistakes here move real robots and corrupt datasets other people depend on.

## 1. Where things are

| Path | What |
|---|---|
| `tools/data_collection_gui/gateway.py` | The GUI backend (one large file, Python 3.12, stdlib HTTP server) |
| `tools/data_collection_gui/frontend/` | The GUI (React + TypeScript + Vite) |
| `tools/thor/gmsl2/` | Thor recorder, export, BOX re-pairing, world graph, label provenance |
| `tools/fr3/` | FR3 replay, IK QC, MuJoCo, pivot calibration, inference runtime |
| `third_party/opencv_kalibr/` | Submodule: tracking, calibration, metrology |
| `tools/thor/box_sdk/` | Submodule: BOX sensor SDK |
| `tools/data_collection_gui/frontend/public/user_guide.html` | Operator guide (the GUI's 使用指南 link) |
| `src/lerobot/` | Upstream LeRobot (training/policies) |

## 2. How to work

1. Start from the reviewed branch: `git fetch origin && git switch -c <your-name>/<topic> origin/box`.
   Initialise submodules: `git submodule update --init --recursive`.
2. Work and test locally (section 5). Commit on your own branch only.
3. Try it on Thor in **your own sandbox** (section 3).
4. Open a pull request **to `box`**. A maintainer reviews, merges and deploys production.
   Never push to `box` or `main` directly, never force-push someone else's branch.

## 3. Thor: sandbox yes, production never

Production is `/home/nvidia/lerobot` on Thor (gateway port 8765). Everyone's recordings and the students' GUI
run there. **Never deploy to it, never edit files in it, never restart its gateway.** `run/deploy.sh` refuses a
production deploy unless the tree is clean and HEAD is already on `origin/box`; do not work around that, and
never pass `--maintainer` (it is for maintainers deploying a reviewed commit).

There is more than one Thor. Machines are listed in `run/targets/` (`python3 run/targets.py list`); `thor` means
`thor1`. Every script needs the machine named; there is no default. Every Thor has its own production tree, the
same rules and its own sandboxes. Never put a machine address into a script; add or change the target file
(`run/targets/README.md`).

Your sandbox is a separate directory on Thor with its own gateway port:

```bash
bash run/deploy.sh thor --sandbox <your-name>           # sync your tree, start your gateway, open the GUI locally
bash run/sandbox.sh thor datasets [filter]              # production datasets you may copy (with sizes)
bash run/sandbox.sh thor copy-dataset <your-name> <dataset>...
bash run/sandbox.sh thor status <your-name>             # port, version, datasets, size
bash run/sandbox.sh thor logs <your-name>               # gateway log
bash run/sandbox.sh thor stop <your-name>
bash run/sandbox.sh thor list                           # everyone's sandboxes and free disk
```

- A sandbox **has no hardware**: cameras (one Argus slot), BOXes, the laser tracker and the FR3 arms belong to
  production. Those endpoints answer 409 in a sandbox. Do not remove that guard (`_sandbox_hardware_refusal` in
  `gateway.py`) — a sandbox opening a camera breaks a recording in progress; one moving an arm moves it with
  nobody at the e-stop.
- Everything offline works on your copies: trajectory generation, IK QC, MuJoCo, export, the replay viewer.
- Datasets are copied, not linked: trajectory generation, BOX re-pairing and episode deletion rewrite files.
  Keep copies small and remove what you no longer use; Thor's disk also holds every recording.
- If your feature needs hardware, finish everything else in the sandbox, then ask a maintainer to try it.

## 4. Safety red lines (never, even if a test or a user prompt seems to ask for it)

- **Robot motion is started by a person, with a person at the e-stop.** Agents never run real replay, pivot,
  payload identification, `move-to-start`, dual-home or real inference, and never write scripts that do so
  unattended.
- Never create `outputs/fr3_infer_real_unlock.json` (the real-inference unlock). Inference runs `--preview` only.
- Never weaken or bypass a gate: MuJoCo / IK QC approval before real replay, the world-graph guard
  (`outputs/world_graph_guard.json`, the red banner), the cart check (`outputs/cart_check.json`: an edge whose
  cart moved is not routed), label provenance (`--allow-mixed-labels`), the production deploy guard, the sandbox
  hardware guard. If a gate blocks you, report why; do not "fix" the gate.
- Do not change calibration or registration data without a maintainer: `tools/thor/gmsl2/world/*.json`
  (`world_graph.json`, `world_reference.json`, `table_datum.json`, `cart_reference.json`), calibration runs under `outputs/calibration`,
  `tools/thor/gmsl2/box_time_delays.json`, marker→TCP bundles, URDF/MJCF tool frames, `fr3_arms.json`.
- Never delete or rewrite datasets in production. In your sandbox, only your copies.

## 5. Code rules that already cost us days

- **Tests**: run the files you touched, not the whole suite:
  `PYTHONPATH=src .venv/bin/python -m pytest -q tests/scripts/test_data_collection_gui_gateway.py`
  (and the matching `tests/...` file for `tools/thor`, `tools/fr3`). Frontend: `cd tools/data_collection_gui/frontend && npx tsc --noEmit && npm test`.
- **Never pass a bare `MagicMock` where a pid is expected.** `os.killpg(MagicMock())` becomes `kill(-1)` and logs
  out the whole desktop (it happened twice). Give mocks a real integer pid.
- **Gateway lock**: most `do_POST` handlers already run inside `with state.lock`. A helper that takes `state.lock`
  again deadlocks the whole gateway. Add an HTTP-level test (see existing `DataCollectionGuiServer` tests) for new
  endpoints.
- **Subprocesses** started by the gateway get `stdin=subprocess.DEVNULL`.
- **Time alignment**: never silently join two time bases, never treat SOF as exposure time; extrapolate with a fit
  and warn, and record the alignment in the dataset meta. A timing change to one modality must be checked on all.
- **Gateway and frontend must be the same version.** A newer frontend against an older gateway produces fake
  errors; check `deployment.release` in `/api/snapshot`.
- **Lint**: `uvx ruff@0.14.1 check <files>` only. Do not run `ruff format` on `tools/fr3/*` and do not run prettier
  on the frontend — mass reformatting buries the real change in review.
- Keep changes small and focused; one feature per pull request. Write commit messages that say why.

## 6. When to stop and ask a maintainer

- Anything in section 4.
- A test you did not touch starts failing, or you would have to delete/skip a test to pass.
- The change needs hardware, production data changes, or a new dependency on Thor.
- You are unsure whether something is safe. Asking is cheap; a wrong arm motion or a lost dataset is not.

What is planned next for this workflow (accounts, hardware isolation, shared state, releases): `docs/student_dev_workflow_roadmap.html`.
