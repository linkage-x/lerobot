"""Load a relative-TCP checkpoint and predict with it: shared by training's reload check and Rollout.

A checkpoint is ``.../checkpoints/<step>/pretrained_model`` as lerobot_train writes it, plus
``fr3_policy_contract.json`` and ``fr3_provenance.json`` (train_relative_tcp.py). Without the contract
nothing says what the inputs are, so :func:`load_checkpoint` refuses it.

:func:`observation_sample` builds the policy's input from raw measurements (TCP poses at the contract's
history offsets, gripper width, wrench, camera frames) with the same functions training used on the
dataset (relative_tcp.state_vector, relative_tcp.crop_resize); offline prediction on a recording and
online preview from the recorder both go through it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.fr3 import relative_tcp as rt

#: Policies that sample their chunk from noise (flow matching). They are given the same noise on every call,
#: so a prediction is a function of the observation: the reload check, offline against online and a preview
#: replay compare exactly, as they do for ACT.
NOISE_POLICIES = ("pi05", "pi0")
NOISE_SEED = 0

#: Real-time chunking (Black, Galliker, Levine 2025; src/lerobot/policies/rtc): a flow-matching policy's new
#: chunk is guided, at every denoising step, toward the part of the previous one the arm will still execute,
#: so a takeover does not jump. Inference only, no training. The values are the ones lerobot_workstation ran
#: pi0.5 with on the real FR3 for two months (fr3_act_infer_real_runtime: weight 10, EXP).
RTC_MAX_GUIDANCE_WEIGHT = 10.0
#: A sample carries the previous chunk as these keys (relative_tcp_policy.with_rtc_prefix): physical actions
#: re-anchored at this observation's TCP, the steps frozen (the inference delay), and the step where guidance
#: has faded to 0 (the steps the new chunk will execute should all be guided: lerobot_workstation ran 9
#: executed steps unguided for weeks when its queue and its horizon disagreed).
RTC_PREFIX_KEY = "rtc.prefix"
RTC_DELAY_KEY = "rtc.delay"
RTC_HORIZON_KEY = "rtc.horizon"

#: Host RAM a pi0 / pi0.5 load peaks at, per byte of its base weights file. Measured 2026-11-17 on the A100:
#: pi05_base (a 13.5 GiB fp32 safetensors) peaked at 41.2 GB RSS, because the pi_gemma wrappers build every
#: layer twice in fp32 before the weights are read. The same load on a workstation with ~10 GB free swapped
#: for ten minutes and froze the desktop until the OOM killer ended it, so a load that does not fit is refused.
LOAD_PEAK_PER_WEIGHT_BYTE = 3.1

#: Policies :func:`load_checkpoint` builds on the meta device and fills tensor by tensor (:func:`load_lean`)
#: instead of through ``from_pretrained``. That peak comes from construction, not from the weights: the pi_gemma
#: wrappers call their parents' ``__init__`` first, so the language model's layers exist four times over in
#: fp32 before one set is kept (PaliGemmaForConditionalGeneration, PaliGemmaModel inside it, GemmaModel inside
#: PiGemmaModel, PiGemmaModel's own layers). On the meta device none of them holds memory; the kept tensors
#: are allocated once, on the target device, in their final dtypes.
LEAN_POLICIES = ("pi05",)

#: Host RAM a lean load (:func:`load_lean`) is allowed to need, the whole process included (Python, torch, the
#: CUDA context's host side, the largest single tensor read from the weights file). Measured 2026-11-23 on the
#: A100 for pi05_base + a LoRA adapter: anonymous RSS peaked at 1.49 GiB (from_pretrained: 38.98 GiB), load
#: 36 s (158 s), GPU 8.82 GiB either way. The weights file is read through mmap, i.e. page cache, which the
#: kernel can drop instead of swapping. 4 GiB leaves room for a larger adapter and the processors.
LEAN_LOAD_PEAK_BYTES = 4 * 2**30


#: Total GPU memory pi0 / pi0.5 needs: inference alone peaked at 8.83 GiB allocated (A100, 2026-11-17: weights
#: 8.71 GiB, one 32-step chunk), plus the CUDA context. An 8 GB card (the RTX 4060) can never hold it, whatever
#: its machine's RAM, so it is refused outright rather than left to fail halfway or swap.
PI05_MIN_GPU_GIB = 10.0


class HostMemoryError(RuntimeError):
    """The machine does not have the free RAM a policy load peaks at."""


class GpuMemoryError(RuntimeError):
    """The GPU is too small for the policy at all."""


def gpu_too_small(policy_type: str, gpu_name: str, total_gib: float) -> str | None:
    """Why ``policy_type`` cannot run on this GPU, or None when it can (or the policy is small)."""
    if policy_type not in NOISE_POLICIES or total_gib >= PI05_MIN_GPU_GIB:
        return None
    return (
        f"{policy_type} cannot run on {gpu_name} ({total_gib:.1f} GiB): it needs a GPU with at least "
        f"{PI05_MIN_GPU_GIB:.0f} GiB. Measured 2026-11-17: inference alone peaks at 8.8 GiB of GPU memory "
        "(weights 8.7 GiB), and loading the model also peaks at about 41 GB of RAM, which froze a workstation. "
        "Run pi0.5 on the A100 (or a GPU with >= 10 GiB); ACT fits on this GPU."
    )


def check_gpu_memory(policy_type: str, device: str) -> None:
    """Refuse a pi0 / pi0.5 load onto a GPU that cannot hold it (the 8 GB RTX 4060)."""
    if policy_type not in NOISE_POLICIES or not str(device).startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(torch.device(device).index or 0)
    reason = gpu_too_small(policy_type, props.name, props.total_memory / 2**30)
    if reason:
        raise GpuMemoryError(reason)


def host_memory_available() -> int | None:
    """MemAvailable in bytes (RAM only: a load that needs swap is the freeze this guards against)."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def weights_file(name_or_path: str | Path) -> Path | None:
    """The ``model.safetensors`` of a checkpoint directory or of a Hugging Face id in the local cache."""
    local = Path(name_or_path) / "model.safetensors"
    if local.is_file():
        return local
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(str(name_or_path), "model.safetensors")
    except Exception:  # noqa: BLE001 -- not a repo id, or no hub library: nothing to measure
        return None
    return Path(cached) if isinstance(cached, str) and Path(cached).is_file() else None


def require_weights(name_or_path: str | Path) -> Path:
    """The pi0 / pi0.5 base weights, or FileNotFoundError.

    PI05Policy.from_pretrained does not raise when it cannot find its weights: it prints a line and returns a
    randomly initialised model (2026-11-17 on the A100, whose cache lacked refs/main, all four ranks did).
    Training or rolling that out is silent nonsense, so the weights are looked up first.
    """
    found = weights_file(name_or_path)
    if found is None:
        raise FileNotFoundError(
            f"the pi0.5 base weights {name_or_path} are not on this machine (a checkpoint directory, or the "
            "Hugging Face cache with its refs/main); without them the model would start from random weights"
        )
    return found


def check_host_memory(
    policy_type: str,
    weights: Path | None,
    processes: int = 1,
    available: int | None = None,
    *,
    lean: bool = False,
) -> None:
    """Refuse a pi0 / pi0.5 load (``processes`` of them at once) that would not fit in free RAM.

    ``lean``: the load goes through :func:`load_lean`, whose peak does not scale with the model's construction.
    """
    if policy_type not in NOISE_POLICIES or weights is None:
        return
    available = host_memory_available() if available is None else available
    if available is None:
        return
    per_load = LEAN_LOAD_PEAK_BYTES if lean else int(weights.stat().st_size * LOAD_PEAK_PER_WEIGHT_BYTE)
    need = per_load * max(1, processes)
    if need > available:
        each = f" ({processes} processes)" if processes > 1 else ""
        raise HostMemoryError(
            f"loading {policy_type}{each} peaks at about {need / 2**30:.0f} GiB of RAM but only "
            f"{available / 2**30:.0f} GiB is available on this machine; it would swap and freeze it. "
            "Free memory or use a machine with more RAM."
        )


def _config(policy):
    return getattr(policy, "config", None) or policy.get_base_model().config


def fixed_noise(policy, device) -> Any:
    import torch

    cfg = _config(policy)
    generator = torch.Generator(device="cpu").manual_seed(NOISE_SEED)
    noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=generator, dtype=torch.float32)
    return noise.to(device)


def _action_normalizer(postprocessor):
    """The postprocessor's unnormaliser step: its stats map physical actions into the policy's space."""
    for step in getattr(postprocessor, "steps", []):
        if hasattr(step, "_normalize_action"):
            return step
    raise ValueError("the checkpoint's postprocessor has no action (un)normaliser")


def normalize_actions(postprocessor, actions: np.ndarray, device) -> Any:
    """Physical (T, 10) actions -> the policy's normalised action space (the inverse of its postprocessor)."""
    import torch

    tensor = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=device)
    return _action_normalizer(postprocessor)._normalize_action(tensor, inverse=False)


def enable_rtc(policy) -> bool:
    """Give a flow-matching policy its RTC processor (unguided predictions are unchanged: without a prefix the
    guided denoiser returns the plain velocity). False for a policy without one (ACT)."""
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    if _config(policy).type not in NOISE_POLICIES:
        return False
    base = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    base.config.rtc_config = RTCConfig(enabled=True, max_guidance_weight=RTC_MAX_GUIDANCE_WEIGHT,
                                       prefix_attention_schedule=RTCAttentionSchedule.EXP)
    base.init_rtc_processor()
    return True


def predict_chunk(policy, preprocessor, postprocessor, sample: dict[str, Any]) -> np.ndarray:
    """(horizon, 10) unnormalised actions for one sample (tensors without a batch dimension). A sample with
    an RTC prefix (:func:`with_rtc_prefix`) is guided toward it; a policy without RTC refuses one."""
    import torch

    batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key.startswith("observation.") and isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
    batch["task"] = [sample.get("task") or ""]
    batch = preprocessor(batch)
    kwargs = {}
    device = next(policy.parameters()).device
    if _config(policy).type in NOISE_POLICIES:
        kwargs["noise"] = fixed_noise(policy, device)
    prefix = sample.get(RTC_PREFIX_KEY)
    if prefix is not None:
        base = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
        rtc = getattr(base.config, "rtc_config", None)
        if rtc is None or not rtc.enabled:
            raise ValueError("an RTC prefix was given to a policy without real-time chunking")
        kwargs["prev_chunk_left_over"] = normalize_actions(postprocessor, np.asarray(prefix), device)
        kwargs["inference_delay"] = int(sample[RTC_DELAY_KEY])
        kwargs["execution_horizon"] = int(sample[RTC_HORIZON_KEY])
    with torch.no_grad():  # the guidance takes its own gradient (torch.enable_grad inside the RTC step)
        actions = policy.predict_action_chunk(batch, **kwargs)
    actions = postprocessor(actions)
    return actions[0].detach().float().cpu().numpy()


def rtc_prefix(T_obs: np.ndarray, targets: np.ndarray, widths: np.ndarray) -> np.ndarray:  # noqa: N803
    """The previous chunk as the new one's RTC prefix: the poses the arm will be commanded to at this
    observation's steps 1, 2, ... (``targets``, (L, 4, 4) in any one frame with ``T_obs``), re-anchored at
    the TCP of this observation, and their widths -- in the actions' own physical units.

    The actions are poses relative to the TCP at the observation, so the previous chunk's own actions are in
    another frame (its own observation's); lerobot_workstation's per-step deltas needed no re-anchoring."""
    return rt.action_chunk(T_obs, targets, widths)


def with_rtc_prefix(sample: dict[str, Any], prefix: np.ndarray, *, delay: int, horizon: int) -> dict[str, Any]:
    """``sample`` plus an RTC prefix: ``delay`` steps frozen (they run while this chunk is predicted), guidance
    fading to 0 at step ``horizon`` (capped at the prefix's length by the RTC step)."""
    prefix = np.asarray(prefix, dtype=np.float32)
    if prefix.ndim != 2 or prefix.shape[1] != rt.POSE_DIM + 1 or not len(prefix):
        raise ValueError(f"RTC prefix: (steps, {rt.POSE_DIM + 1}) expected, got {prefix.shape}")
    if not 0 <= delay <= horizon:
        raise ValueError(f"RTC: delay {delay} and horizon {horizon} must satisfy 0 <= delay <= horizon")
    return {**sample, RTC_PREFIX_KEY: prefix, RTC_DELAY_KEY: int(delay), RTC_HORIZON_KEY: int(horizon)}


@dataclass
class LoadedPolicy:
    path: Path
    contract: rt.PolicyContract
    provenance: dict[str, Any]
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: str
    rtc: bool = False  # real-time chunking available (a flow-matching policy): predict takes an RTC prefix

    def predict(self, sample: dict[str, Any]) -> np.ndarray:
        return predict_chunk(self.policy, self.preprocessor, self.postprocessor, sample)


def _recompute_unsaved_buffers(policy) -> None:
    """Fill the buffers a weights file does not hold (``persistent=False``) as the usual build leaves them.

    The usual build makes them on the CPU in float32 and casts them with the rest of the model
    (``to_bfloat16_for_selected_params``), so they are computed on the CPU here and cast to the dtype the meta
    build gave them. A buffer no rule here knows is refused: left as ``to_empty`` made it, it is garbage.
    """
    import torch

    saved = set(policy.state_dict().keys())
    for name, module in policy.named_modules():
        unsaved = sorted(b for b, _ in module.named_buffers(recurse=False) if f"{name}.{b}".lstrip(".") not in saved)
        if not unsaved:
            continue
        kind = type(module).__name__
        if kind.endswith("RotaryEmbedding") and set(unsaved) <= {"inv_freq", "original_inv_freq"}:
            fresh = type(module)(config=module.config, device="cpu")
            module.attention_scaling = fresh.attention_scaling
            values = {b: getattr(fresh, b) for b in unsaved}
        elif kind == "SiglipVisionEmbeddings" and unsaved == ["position_ids"]:
            values = {"position_ids": torch.arange(module.num_positions).expand((1, -1))}
        else:
            raise RuntimeError(
                f"{name} ({kind}) has buffers {unsaved} that the weights file does not hold and that no rule here "
                "computes; load this checkpoint with lean=False"
            )
        for b, value in values.items():
            buf = getattr(module, b)
            if tuple(buf.shape) != tuple(value.shape):
                raise RuntimeError(f"{name}.{b}: computed {tuple(value.shape)}, the model has {tuple(buf.shape)}")
            buf.copy_(value.to(buf.dtype))


def _load_weights_streaming(policy, weights: Path) -> None:
    """``from_pretrained``'s state-dict load, one tensor at a time (strict: every key, nothing extra).

    The key mapping is the policy's own ``_fix_pytorch_state_dict_keys`` plus the ``model.`` prefix, run on the
    names only; it copies ``lm_head`` into ``embed_tokens`` with ``value.clone()``, which a name answers.
    """
    from safetensors import safe_open

    class _Name(str):
        def clone(self) -> _Name:
            return self

    with safe_open(str(weights), framework="pt", device="cpu") as f:
        fixed = policy._fix_pytorch_state_dict_keys({k: _Name(k) for k in f.keys()}, policy.config)  # noqa: SLF001, SIM118
        source = {(k if k.startswith("model.") else f"model.{k}"): str(v) for k, v in fixed.items()}
        targets = policy.state_dict(keep_vars=True)
        missing, unexpected = sorted(set(targets) - set(source)), sorted(set(source) - set(targets))
        if missing or unexpected:
            raise RuntimeError(
                f"{weights} does not match the model: {len(missing)} missing {missing[:5]}, "
                f"{len(unexpected)} unexpected {unexpected[:5]}"
            )
        for key, target in targets.items():
            value = f.get_tensor(source[key])
            if tuple(value.shape) != tuple(target.shape):
                raise RuntimeError(f"{key}: {tuple(value.shape)} in {weights}, {tuple(target.shape)} in the model")
            target.copy_(value)
            del value


def load_lean(cfg, weights: Path, device: str):
    """The policy ``from_pretrained(weights, config=cfg)`` returns, without its construction peak in RAM.

    Built on the meta device (no memory), allocated once on ``device`` in the dtypes the build chose, the
    unsaved buffers recomputed, then the weights copied in one tensor at a time. The result is bit-identical
    to ``from_pretrained``'s (A100, 2026-11-23: all 1074 tensors and buffers equal, plain and RTC-guided chunks
    equal to the bit); peak host RAM: :data:`LEAN_LOAD_PEAK_BYTES`.
    """
    import torch

    from lerobot.policies.factory import get_policy_class

    cls = get_policy_class(cfg.type)
    cfg.device = "meta"  # the policy's __init__ moves its model to cfg.device
    try:
        with torch.device("meta"):
            policy = cls(cfg)
    finally:
        cfg.device = device
    policy.to_empty(device=device)
    with torch.no_grad():
        _recompute_unsaved_buffers(policy)
        _load_weights_streaming(policy, weights)
    left = [n for n, t in [*policy.named_parameters(), *policy.named_buffers()] if t.is_meta]
    left += [f"{n}.{a}" for n, m in policy.named_modules() for a, v in vars(m).items()
             if isinstance(v, torch.Tensor) and v.is_meta]
    if left:
        raise RuntimeError(f"tensors still on the meta device after a lean load: {left[:5]}")
    return policy


def load_checkpoint(path: Path, device: str | None = None, *, lean: bool = True) -> LoadedPolicy:
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    path = Path(path)
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    from tools.fr3 import relative_tcp_remote

    if relative_tcp_remote.is_stub(path):  # weights on a policy server (pi0.5 is not run on Thor)
        return relative_tcp_remote.load_remote(path)  # type: ignore[return-value]
    contract = rt.PolicyContract.read(path)
    prov_path = path / rt.PROVENANCE_FILE
    provenance = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    check_gpu_memory(cfg.type, device)  # before anything is loaded: the 4060 is refused, not tried
    lean = lean and cfg.type in LEAN_POLICIES

    def load(weights_dir: str | Path):
        weights = require_weights(weights_dir) if cfg.type in NOISE_POLICIES else None
        check_host_memory(cfg.type, weights, lean=lean)
        if lean:
            return load_lean(cfg, weights, device)
        return get_policy_class(cfg.type).from_pretrained(weights_dir, config=cfg)

    if getattr(cfg, "use_peft", False):
        # A LoRA run saves the adapter only; the base it adapted is named in the adapter's config and has to
        # be on this machine (the Hugging Face cache; nothing is downloaded here).
        from peft import PeftConfig, PeftModel

        adapter = PeftConfig.from_pretrained(str(path))
        policy = PeftModel.from_pretrained(load(adapter.base_model_name_or_path), str(path), config=adapter)
    else:
        policy = load(path)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    expected = {c.key for c in contract.cameras} | {"observation.state"}
    got = set(cfg.input_features)
    if got != expected:
        raise ValueError(f"{path}: the policy's inputs {sorted(got)} are not the contract's {sorted(expected)}")
    loaded = LoadedPolicy(path, contract, provenance, policy, pre, post, device)
    loaded.rtc = enable_rtc(policy)
    return loaded


def observation_sample(
    contract: rt.PolicyContract,
    history_poses: np.ndarray,
    width: float,
    images: dict[str, np.ndarray],
    wrench: np.ndarray | None = None,
    *,
    already_cropped: bool = False,
    task: str = "",
) -> dict[str, Any]:
    """The policy's input from measurements.

    ``history_poses``: (history + 1, 4, 4) TCP poses at ``contract.history_offsets`` (last = now), in any one
    frame; ``images``: camera name -> HxWx3 uint8 RGB, the full source frame unless ``already_cropped``.
    """
    import torch

    sample: dict[str, Any] = {
        "observation.state": torch.from_numpy(rt.state_vector(contract, history_poses, width, wrench)),
        "task": task or contract.task,
    }
    for cam in contract.cameras:
        if cam.name not in images:
            raise ValueError(f"no frame from {cam.name}")
        frame = images[cam.name] if already_cropped else rt.crop_resize(images[cam.name], cam)
        if frame.shape != (cam.height, cam.width, 3):
            raise ValueError(f"{cam.name}: frame {frame.shape}, expected {(cam.height, cam.width, 3)}")
        sample[cam.key] = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
    return sample
