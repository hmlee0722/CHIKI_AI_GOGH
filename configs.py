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
#EXPERIMENT
# -------------------------------------------------------------------------
cfg.TARGET_BLOCKS = ["up_blocks.0.attentions.1"]
# target blocks to apply IP-Adapter
cfg.IP_ADAPTER_SCALE = 1.0
# scale for style strength
cfg.CONTROLNET_CONDITIONING_SCALE = 0.8
# scale for controlling structure of content image
cfg.SHORT_SIDE = 768
# short side size for resizing content image before feeding into controlnet
cfg.STYLE_IMAGE_ID = 103
# default style image id
cfg.PROMPT = "masterpiece, best quality, high quality, beautiful, adorable, cute, perfect face, soft smiling face, perfect eyes"
# default prompt for generation
cfg.NEGATIVE_PROMPT = "text, watermark, lowres, worst quality, low quality, blurry, deformed, noisy, saturationm"
# default negative prompt for generation
cfg.SEED = 42