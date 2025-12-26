import torch
from yacs.config import CfgNode as CN

cfg = CN()
# -------------------------------------------------------------------------
#MODEL
# -------------------------------------------------------------------------
cfg.SDXL_BASE_MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
# diffusion model
cfg.CONTROLNET_PATH = "xinsir/controlnet-tile-sdxl-1.0"
# tile controlnet for controlling structure of content image
cfg.IP_ADAPTER_EXTRACTOR_PATH = "IP-Adapter/sdxl_models/image_encoder"
# image encoder of IP-Adapter
cfg.IP_ADAPTER_MODULE_PATH = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
# cross attention module of IP-Adapter

# -------------------------------------------------------------------------
#ENVIRONMENT
# -------------------------------------------------------------------------
cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# device setting
cfg.DTYPE = torch.float16
# data type for model

# -------------------------------------------------------------------------
#EXPERIMENT
# -------------------------------------------------------------------------
cfg.TARGET_BLOCKS = ["up_blocks.0.attentions.1"]
# target blocks to apply IP-Adapter
cfg.IP_ADAPTER_SCALE = 1.2
# scale for style strength
cfg.STYLE_IMAGE_ID = 103
# default style image id
# 103 : starry night