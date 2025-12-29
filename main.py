import torch
import io
import time
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
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
        controlnet_cond_image = resize_image(content_image, short=1024)

        kwargs = {
            'pil_image' : style_image,
            'image' : controlnet_cond_image
        }

        # 3. GPU Inference
        print("🎨Generating styled image...")
        with torch.no_grad():
            generated = MODEL.generate(
                prompt="masterpiece, best quality, high quality, van gogh style, oil painting",
                negative_prompt="text, watermark, lowres, worst quality, low quality, blurry, deformed, noisy, saturationm",
                guidance_scale=5.0,
                num_samples=1,
                seed=42,
                controlnet_conditioning_scale=0.6,
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
        controlnet_cond_image = resize_image(content_image, short=1024)

        kwargs = {
            'pil_image' : style_image,
            'image' : controlnet_cond_image
        }

        # 3. GPU Inference
        print("🎨Generating styled image...")
        with torch.no_grad():
            generated = MODEL.generate(
                prompt="masterpiece, best quality, high quality, van gogh style, oil painting",
                negative_prompt="text, watermark, lowres, worst quality, low quality, blurry, deformed, noisy, saturationm",
                guidance_scale=5.0,
                num_samples=1,
                seed=42,
                controlnet_conditioning_scale=0.6,
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
    """
    서비스 상태 및 GPU, 모델 로드 정보를 반환합니다.
    """
    try:
        # GPU 메모리 정보 (torch 사용 시)
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated": f"{torch.cuda.memory_allocated(i) / 1024**2:.2f} MB",
                    "memory_reserved": f"{torch.cuda.memory_reserved(i) / 1024**2:.2f} MB",
                })

        info = {
            "status": "healthy",
            "timestamp": time.time(),
            "model_loaded": MODEL is not None,
            "config": {
                "device": str(cfg.DEVICE),
                "dtype": str(cfg.DTYPE),
                "style_image_id": cfg.STYLE_IMAGE_ID,
                "ip_adapter_scale": cfg.IP_ADAPTER_SCALE
            },
            "gpu": {
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "details": gpu_info
            }
        }
        
        # 모델이 로드되지 않았을 경우 상태를 warning으로 표시하고 싶다면 아래 주석 해제
        # if MODEL is None:
        #     info["status"] = "degraded"
            
        return JSONResponse(content=info, status_code=200)

    except Exception as e:
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)}, 
            status_code=500
        )
