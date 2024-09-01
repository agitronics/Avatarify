import asyncio
import websockets
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from video_processor import GPUAcceleratedVideoProcessor
from camera_manager import CameraManager
from face_swap import load_face_swap_model
from style_transfer import load_style_transfer_model
from voice_cloning import load_voice_cloning_model
from advanced_effects import AdvancedVideoEffects

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_swap_model = load_face_swap_model("path/to/face_swap_model.pth")
style_transfer_model = load_style_transfer_model()
voice_cloning_model = load_voice_cloning_model()
video_processor = GPUAcceleratedVideoProcessor(face_swap_model, style_transfer_model)
camera_manager = CameraManager()
advanced_effects = AdvancedVideoEffects()

class EffectRequest(BaseModel):
    effect: str

@app.get("/effects")
async def get_effects():
    return ["face_swap", "style_transfer", "cartoon", "deep_dream", "glitch", "vhs", "rainbow"]

@app.post("/apply-effect")
async def apply_effect(effect_request: EffectRequest):
    video_processor.set_effect(effect_request.effect)
    return {"message": f"Applied effect: {effect_request.effect}"}

async def video_stream(websocket, path):
    try:
        camera_manager.start()
        while True:
            frame = await camera_manager.get_frame()
            processed_frame = video_processor.process_frame(frame)
            await websocket.send(processed_frame.tobytes())
    finally:
        camera_manager.stop()

if __name__ == "__main__":
    server = websockets.serve(video_stream, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(server)
    uvicorn.run(app, host="0.0.0.0", port=8000)