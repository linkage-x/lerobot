import dataclasses
from pathlib import Path
from shutil import copytree

from huggingface_hub import hf_hub_download


def config_dataclass(cls):
    """`@dataclass`, unless the transformers in use already applied it.

    transformers>=5 makes `PretrainedConfig` a dataclass and runs `@dataclass` over every
    subclass from `__init_subclass__`. Applying it a second time re-reads the class body, where
    the `field(init=False)` sentinels have already been consumed -- so the fields come back as
    ordinary init arguments without defaults, land after the ten defaulted ones the base now
    contributes, and the class raises `TypeError: non-default argument 'backbone_cfg' follows
    default argument` at import time.

    That is not a Gr00t-only failure: `lerobot/policies/__init__.py` imports Gr00t eagerly, so
    the exception takes down `lerobot.policies` -- and with it `lerobot.policies.factory`, which
    `lerobot_train` imports before it can train anything at all, pi0.5 included.

    Under transformers 4 `PretrainedConfig` is a plain class, nothing has been applied, and the
    decorator is still required. Hence the check rather than a removal: `is_dataclass` is False
    exactly in the case where the decorator is still doing work.
    """
    return cls if dataclasses.is_dataclass(cls) else dataclasses.dataclass(cls)


def ensure_eagle_cache_ready(vendor_dir: Path, cache_dir: Path, assets_repo: str) -> None:
    """Populate the Eagle processor directory in cache and ensure tokenizer assets exist.

    - Copies the vendored Eagle files into cache_dir (overwriting when needed).
    - Downloads vocab.json and merges.txt into the same cache_dir if missing.
    """
    cache_dir = Path(cache_dir)
    vendor_dir = Path(vendor_dir)

    try:
        # Populate/refresh cache with vendor files to ensure a complete processor directory
        print(f"[GROOT] Copying vendor Eagle files to cache: {vendor_dir} -> {cache_dir}")
        copytree(vendor_dir, cache_dir, dirs_exist_ok=True)
    except Exception as exc:  # nosec: B110
        print(f"[GROOT] Warning: Failed to copy vendor Eagle files to cache: {exc}")

    required_assets = [
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.json",
        "special_tokens_map.json",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ]

    print(f"[GROOT] Assets repo: {assets_repo} \n Cache dir: {cache_dir}")

    for fname in required_assets:
        dst = cache_dir / fname
        if not dst.exists():
            print(f"[GROOT] Fetching {fname}")
            hf_hub_download(
                repo_id=assets_repo,
                filename=fname,
                repo_type="model",
                local_dir=str(cache_dir),
            )
