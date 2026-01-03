import torch
import io
import time
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, DDIMScheduler
from ip_adapter import IPAdapterXL
from utils import resize_image, empty_cache
from configs import cfg

app = FastAPI(title="CHIKI Gogh Style Transfer API",
              description="An API for style transfer using Stable Diffusion XL with ControlNet and IP-Adapter"
             )
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

@app.on_event("startup")
def load_models():
    global MODEL
    print(f"🚀Loading models...")

    # 1. Load ControlNet
    controlnet = ControlNetModel.from_pretrained(cfg.CONTROLNET_PATH, 
                                                 torch_dtype=DTYPE).to(DEVICE)
    
    # 2. Load Pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        cfg.SDXL_BASE_MODEL_PATH,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=DTYPE
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Optimized for memory/speed without needing xformers
    pipe.vae.enable_tiling()
    pipe.enable_attention_slicing()

    # 3. Load IP-AdpaterXL
    MODEL = IPAdapterXL(
        pipe,
        cfg.IP_ADAPTER_EXTRACTOR_PATH,
        cfg.IP_ADAPTER_MODULE_PATH,
        DEVICE,
        target_blocks=cfg.TARGET_BLOCKS
    )

    del pipe, controlnet
    empty_cache()

    print(f"✅Models loaded successfully!")

@app.post("/api/photo/ai_upload")
async def generate_from_upload(
    content_file: UploadFile = File(...)
):
    try:
        # 1. Load Style Image from server
        style_path = f'data/style/{cfg.STYLE_IMAGE_ID}.jpg'
        style_image = Image.open(style_path).convert("RGB")

        # 2. Process Uploaded Content Image
        content_data = await content_file.read()
        content_image = Image.open(io.BytesIO(content_data)).convert("RGB")
        W, H = content_image.size

        # Prepare for SDXL (SDXL works better with larger size inputs)
        controlnet_cond_image = resize_image(content_image, short=cfg.SHORT_SIDE)

        kwargs = {
            'pil_image' : style_image,
            'image' : controlnet_cond_image
        }

        # 3. GPU Inference
        print("🎨Generating styled image...")
        with torch.no_grad():
            generated = MODEL.generate(
                prompt=cfg.PROMPT,
                negative_prompt=cfg.NEGATIVE_PROMPT,
                guidance_scale=5.0,
                num_samples=1,
                seed=cfg.SEED,
                controlnet_conditioning_scale=cfg.CONTROLNET_CONDITIONING_SCALE,
                scale=cfg.IP_ADAPTER_SCALE,
                **kwargs
            )

        # 4. Return Result
        result = generated[0].resize((W, H), resample=Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        buffer.seek(0)
        print("✅Generation complete.")
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        print(f"❌Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/photo/ai")
async def generate(
    request: Request,
):
    try:
        # 1. Load Style Image from server
        style_path = f'data/style/{cfg.STYLE_IMAGE_ID}.jpg'
        style_image = Image.open(style_path).convert("RGB")

        # 2. Process Uploaded Content Image
        content_data = await request.json()
        url = content_data.get("imageUrl")
        response = requests.get(url)
        content_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        W, H = content_image.size

        # Prepare for SDXL (SDXL works better with larger size inputs)
        controlnet_cond_image = resize_image(content_image, short=cfg.SHORT_SIDE)

        kwargs = {
            'pil_image' : style_image,
            'image' : controlnet_cond_image
        }

        # 3. GPU Inference
        print("🎨Generating styled image...")
        with torch.no_grad():
            generated = MODEL.generate(
                prompt=cfg.PROMPT,
                negative_prompt=cfg.NEGATIVE_PROMPT,
                guidance_scale=5.0,
                num_samples=1,
                seed=cfg.SEED,
                controlnet_conditioning_scale=cfg.CONTROLNET_CONDITIONING_SCALE,
                scale=cfg.IP_ADAPTER_SCALE,
                **kwargs
            )
        # 4. Return Result
        result = generated[0].resize((W, H), resample=Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        buffer.seek(0)
        print("✅Generation complete.")
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        print(f"❌Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    '''
    naive health check endpoint
    '''
    return JSONResponse(
        content={"status": "healthy"}, 
        status_code=200
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
