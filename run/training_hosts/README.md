# Training machines

One JSON file per machine the Training page trains on over ssh (the gateway's own machine is always
listed as "This machine"). Like `run/targets/`, addresses live here and nowhere in code.

**This repository is public.** A machine whose address must not be published (a data-centre box reached
through a public IP and port) goes in `<id>.local.json` instead: same keys, git-ignored, and the page
still calls it `<id>`. Copy that file to each computer whose gateway trains there.

| key | meaning |
|---|---|
| `label` | what the page shows |
| `user`, `address`, `port` | how the gateway reaches it (`ssh -p port user@address`); the gateway's machine needs a key there |
| `hostname` | the machine's real hostname, checked over ssh before code is synced into it |
| `repo_dir` | a directory used only for training; the page marks it and syncs this tree into it (rsync --delete) |
| `python` | the interpreter, relative to `repo_dir` or absolute |
| `pythonpath_prefix` | optional, put before `src:.` on PYTHONPATH (e.g. a `sitecustomize` shim) |
| `note` | optional |

The environment in `repo_dir` is set up once by hand:

```bash
cd <repo_dir> && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[pi,peft]" av
```

pi0.5 also needs `lerobot/pi05_base` and the `google/paligemma-3b-pt-224` tokenizer in that machine's
Hugging Face cache: training runs offline (`HF_HUB_OFFLINE=1`).
