"""Load the funnel105 checkpoint the way the rollout does (minus dataset metadata) and time inference."""
import sys, time
from pathlib import Path
import torch
from peft import PeftConfig, PeftModel
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

ckpt = Path(sys.argv[1]).resolve()
cfg = TrainPipelineConfig.from_pretrained(ckpt).policy
cfg.device = "cuda"
cfg.pretrained_path = ckpt
peft_cfg = PeftConfig.from_pretrained(str(ckpt))
print("base", peft_cfg.base_model_name_or_path)
t0 = time.time()
base_dir = sys.argv[2] if len(sys.argv) > 2 else peft_cfg.base_model_name_or_path
print("loading base from", base_dir)
policy = PI05Policy.from_pretrained(base_dir, config=cfg)
policy = PeftModel.from_pretrained(policy, str(ckpt), config=peft_cfg)
policy.to("cuda").eval()
print(f"loaded in {time.time()-t0:.1f}s, vram {torch.cuda.memory_allocated()/1e9:.1f} GB")
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(ckpt),
                                     preprocessor_overrides={"device_processor": {"device": "cuda"}})
print("input_features", {k: tuple(v.shape) for k, v in cfg.input_features.items()})
print("output_features", {k: tuple(v.shape) for k, v in cfg.output_features.items()})
obs = {}
for k, ft in cfg.input_features.items():
    obs[k] = torch.rand(*ft.shape) if k.startswith("observation.images") else torch.zeros(*ft.shape)
obs["task"] = "insert the peg"
batch = pre(obs)
base = policy.get_base_model()
times = []
with torch.inference_mode():
    for _ in range(6):
        torch.cuda.synchronize(); t = time.time()
        chunk = base.predict_action_chunk(batch)
        torch.cuda.synchronize(); times.append(time.time() - t)
print("chunk", tuple(chunk.shape), "latency_s", [round(x, 3) for x in times])
print("PROBE_OK")
