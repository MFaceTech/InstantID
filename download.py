import os
from huggingface_hub import hf_hub_download
local_dir = "checkpoints"
os.makedirs(local_dir, exist_ok=True)

hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir=local_dir)
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir=local_dir)
hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir=local_dir)

