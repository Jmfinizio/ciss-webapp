import os
import cv2
import re
import uuid
import time
import json
import asyncio
import logging
import numpy as np
import pandas as pd
import tempfile
import traceback
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from office365.runtime.auth.client_credential import ClientCredential
from office365.sharepoint.client_context import ClientContext

# Initialize application with safe globals
torch.serialization.add_safe_globals([
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.linear.Linear,
    torch.nn.modules.container.Sequential,
    torch.nn.modules.activation.SiLU,
    torch.nn.modules.container.ModuleList,
    torch.nn.modules.upsampling.Upsample,
    torch.nn.modules.pooling.MaxPool2d
])

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
app.mount("/static", os.path.join(Path(__file__).parent, "static"), name="static")

# Configuration values
MODEL_PATH = os.path.join(Path(__file__).parent.parent, "parents_child_stranger.pt")
POSE_MODEL_PATH = os.path.join(Path(__file__).parent.parent, "yolov8n-pose.pt")
MAX_VIDEO_SIZE = 500 * 1024 * 1024
OUTPUT_DIR = os.path.join(Path(__file__).parent, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global state
PROGRESS_STORE = {}

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

LANDMARKS = {
    "left_eye": [33, 133, 159, 145, 160, 144],
    "right_eye": [362, 263, 386, 374, 387, 373],
    "left_eyebrow": [70, 63, 105],
    "right_eyebrow": [300, 293, 334],
    "mouth": [13, 14, 78, 308],
    "jaw": [152]
}

@app.on_event("startup")
async def initialize_models():
    """Initialize models with warmup inference"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing models on {device.upper()}")

        # Initialize detection model
        app.state.detection_model = YOLO(MODEL_PATH).to(device)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        app.state.detection_model(dummy, verbose=False)  # Warmup
        
        # Initialize pose model
        app.state.pose_model = YOLO("yolov8n-pose.pt").to(device)
        app.state.pose_model(dummy, verbose=False)  # Warmup
        
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise RuntimeError(f"Model initialization failed: {str(e)}")

def get_sharepoint_context(site_url: str, client_id: str, client_secret: str):
    """Create SharePoint client context"""
    return ClientContext(site_url).with_credentials(
        ClientCredential(client_id, client_secret))

def update_progress(process_id: str, current: int, total: int, message: str):
    """Update progress store with analysis status"""
    PROGRESS_STORE[process_id] = {
        "percent": min(100, (current / total) * 100),
        "message": message,
        "current": current,
        "total": total,
        "status": "processing"
    }

def detect_child_and_crop(frame):
    try:
        results = app.state.detection_model(frame, verbose=False)[0]
        class_ids = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        bboxes = results.boxes.xyxy.cpu().numpy()

        child_bbox = None
        adult_bbox = None
        stranger_bbox = None
        adult_conf = 0
        stranger_conf = 0

        for box, cls, conf in zip(bboxes, class_ids, confidences):
            if conf > 0.6:
                if cls == 1:  # Child
                    child_bbox = box
                elif cls == 0:  # Adult
                    if conf > adult_conf:
                        adult_bbox = box
                        adult_conf = conf
                elif cls == 2:  # Stranger
                    if conf > stranger_conf:
                        stranger_bbox = box
                        stranger_conf = conf

        if child_bbox is None:
            return None, 0, None, None

        def euclidean_distance(box1, box2):
            if box1 is None or box2 is None:
                return None
            center1 = ((box1[0] + box1[2])/2, (box1[1] + box1[3])/2)
            center2 = ((box2[0] + box2[2])/2, (box2[1] + box2[3])/2)
            return np.linalg.norm(np.array(center1) - np.array(center2))

        def calculate_diagonal(bbox):
            return np.sqrt((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2) if bbox else 0

        adjusted_distance_adult = None
        if adult_bbox and adult_conf > 0.6:
            adult_diagonal = calculate_diagonal(adult_bbox)
            child_diagonal = calculate_diagonal(child_bbox)
            raw_dist = euclidean_distance(child_bbox, adult_bbox)
            if raw_dist:
                adjusted_distance_adult = raw_dist * (abs(adult_diagonal - 1.67*child_diagonal)/1000)

        adjusted_distance_stranger = None
        if stranger_bbox and stranger_conf > 0.6:
            stranger_diagonal = calculate_diagonal(stranger_bbox)
            child_diagonal = calculate_diagonal(child_bbox)
            raw_dist = euclidean_distance(child_bbox, stranger_bbox)
            if raw_dist:
                adjusted_distance_stranger = raw_dist * (abs(stranger_diagonal - 1.67*child_diagonal)/1000)

        def categorize(d):
            if d is None: return None
            return 2 if d < 20 else 1 if d < 200 else 0

        category_adult = categorize(adjusted_distance_adult)
        category_stranger = categorize(adjusted_distance_stranger)

        if category_adult is not None:
            if category_adult == 2:
                category_stranger = 0
            elif category_adult == 1:
                category_stranger = 1
            elif category_adult == 0:
                category_stranger = 2
        elif category_stranger is not None:
            if category_stranger == 2:
                category_adult = 0
            elif category_stranger == 1:
                category_adult = 1
            elif category_stranger == 0:
                category_adult = 2

        x1, y1, x2, y2 = map(int, child_bbox)
        child_roi = frame[y1:y2, x1:x2]

        return child_roi, len([x for x in class_ids if x == 0]), category_adult, category_stranger

    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return None, 0, None, None

def process_pose(image):
    try:
        results = app.state.pose_model(image, verbose=False)
        if results and hasattr(results[0], 'keypoints'):
            return results[0].keypoints.xy[0].cpu().numpy()
        return None
    except Exception as e:
        logger.error(f"Pose processing error: {str(e)}")
        return None

def facial_keypoints(image, prev_landmarks=None):
    try:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return 0, None

        current_landmarks = {}
        for key, indices in LANDMARKS.items():
            current_landmarks[key] = [
                (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]))
                for lm in [results.multi_face_landmarks[0].landmark[i] for i in indices]
            ]

        movement_score = 0
        if prev_landmarks:
            total_diff = sum(
                np.sqrt((cx - px)**2 + (cy - py)**2)
                for key in LANDMARKS
                for (px, py), (cx, cy) in zip(prev_landmarks.get(key, []), current_landmarks.get(key, []))
            )
            valid_points = sum(len(landmarks) for landmarks in current_landmarks.values())
            movement_score = 2 if (total_diff/valid_points) > 6 else 1 if (total_diff/valid_points) > 3 else 0

        return movement_score, current_landmarks
    except Exception as e:
        logger.error(f"Facial processing error: {str(e)}")
        return 0, None

def calculate_body_movement(current_pose, previous_pose):
    if current_pose is None or previous_pose is None:
        return 0.0
    
    valid_points = 0
    total_movement = 0.0
    
    for prev, curr in zip(previous_pose, current_pose):
        if not (np.isnan(prev).any() or np.isnan(curr).any()):
            valid_points += 1
            total_movement += np.linalg.norm(curr - prev)
    
    return round(total_movement / valid_points, 2) if valid_points > 0 else 0.0

async def download_sharepoint_file(file_id: str, site_url: str, client_id: str, client_secret: str, temp_dir: Path):
    ctx = get_sharepoint_context(site_url, client_id, client_secret)
    file = ctx.web.get_file_by_id(file_id)
    ctx.load(file)
    ctx.execute_query()
    
    video_path = temp_dir / file.properties["Name"]
    with open(video_path, "wb") as f:
        file.download(f).execute_query()
    
    return video_path

async def process_video_async(process_id: str, video_path: Path, session_dir: Path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = int(total_frames / original_fps)
        
        results = []
        prev_landmarks = None
        prev_pose = None

        PROGRESS_STORE[process_id] = {
            "percent": 0,
            "message": "Starting analysis...",
            "current": 0,
            "total": duration_seconds,
            "status": "processing"
        }

        for second in range(duration_seconds):
            if PROGRESS_STORE.get(process_id, {}).get("status") == "cancelled":
                logger.info(f"Process {process_id} cancelled")
                break

            current_frame = int(second * original_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Skipping frame {current_frame}")
                continue

            timecode = f"{second//60:02d}:{second%60:02d}"
            update_progress(process_id, second+1, duration_seconds, f"Processing {timecode}")

            try:
                child_roi, _, adult_dist, stranger_dist = detect_child_and_crop(frame)
                if child_roi is None:
                    continue

                face_score, curr_landmarks = facial_keypoints(child_roi, prev_landmarks)
                prev_landmarks = curr_landmarks

                pose_kps = process_pose(child_roi)
                body_movement = calculate_body_movement(pose_kps, prev_pose)
                prev_pose = pose_kps

                results.append({
                    "second": second,
                    "timecode": timecode,
                    "face_movement": face_score,
                    "body_movement": body_movement,
                    "adult_distance": adult_dist if adult_dist is not None else -1,
                    "stranger_distance": stranger_dist if stranger_dist is not None else -1
                })

                if second % 30 == 0:
                    pd.DataFrame(results).to_csv(session_dir / "partial.csv", index=False)

            except Exception as e:
                logger.error(f"Error processing second {second}: {str(e)}")

        pd.DataFrame(results).to_csv(session_dir / "analysis.csv", index=False)
        PROGRESS_STORE[process_id]["status"] = "completed"
        PROGRESS_STORE[process_id]["result"] = str(session_dir / "analysis.csv")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        PROGRESS_STORE[process_id]["status"] = "error"
        PROGRESS_STORE[process_id]["error"] = str(e)
    finally:
        cap.release()
        if video_path.exists():
            video_path.unlink()

@app.post("/api/process-video")
async def start_processing(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(None),
    file_id: str = Form(None),
    site_url: str = Form(None),
    client_id: str = Form(None),
    client_secret: str = Form(None)
):
    process_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.mkdtemp())
    session_dir = OUTPUT_DIR / f"session_{process_id}"
    session_dir.mkdir(exist_ok=True)

    try:
        if file_id:
            video_path = await download_sharepoint_file(file_id, site_url, client_id, client_secret, temp_dir)
        else:
            video_path = temp_dir / video.filename
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)

        background_tasks.add_task(process_video_async, process_id, video_path, session_dir)
        return JSONResponse({"process_id": process_id, "status_url": f"/api/progress/{process_id}"})

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(500, detail=str(e))

@app.get("/api/progress/{process_id}")
async def get_progress(process_id: str):
    progress = PROGRESS_STORE.get(process_id)
    if not progress:
        raise HTTPException(404, detail="Process not found")
    if progress.get("status") == "error":
        raise HTTPException(500, detail=progress.get("error", "Unknown error"))
    return progress

@app.post("/api/cancel/{process_id}")
async def cancel_processing(process_id: str):
    if process_id in PROGRESS_STORE:
        PROGRESS_STORE[process_id]["status"] = "cancelled"
        return {"status": "cancellation_requested"}
    raise HTTPException(404, detail="Process not found")

@app.get("/api/results/{process_id}")
async def get_results(process_id: str):
    progress = PROGRESS_STORE.get(process_id)
    if not progress:
        raise HTTPException(404, detail="Process not found")
    
    if progress.get("status") != "completed":
        raise HTTPException(425, detail="Analysis not complete")
    
    result_path = Path(progress.get("result", ""))
    if not result_path.exists():
        raise HTTPException(404, detail="Results not found")
    
    return FileResponse(result_path, filename="analysis.csv")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # Avoid API and static file paths
    if full_path.startswith(("api/", "static/")):
        raise HTTPException(status_code=404)
    
    # Serve index.html for all other routes.
    frontend_path = Path(os.path.join(Path(__file__).parent.parent, "frontend", "index.html"))
    if not frontend_path.exists():
        raise HTTPException(status_code=404, detail=f"Frontend not found at {frontend_path.absolute()}")
    return FileResponse(frontend_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)