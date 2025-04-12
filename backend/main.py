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
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect, Bottleneck, Concat, DFL
from torch.nn import Conv2d, BatchNorm2d, Linear, SiLU, ModuleList, Upsample, MaxPool2d

# Initialize application with safe globals
torch.serialization.add_safe_globals([
    Conv2d, BatchNorm2d, Linear, Sequential, SiLU, ModuleList, Upsample, MaxPool2d,
    DetectionModel, Conv, C2f, SPPF, Detect, Bottleneck, Concat, DFL
])

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

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Configuration
MODEL_PATH = "parents_child_stranger.pt"
MAX_VIDEO_SIZE = 500 * 1024 * 1024
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global state
PROGRESS_STORE = {}
current_process_id = None
ANALYSIS_ACTIVE = False

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
async def startup_event():
    app.state.detection_model = YOLO(MODEL_PATH).to('cuda' if torch.cuda.is_available() else 'cpu')
    app.state.pose_model = YOLO("yolov8n-pose.pt").to('cuda' if torch.cuda.is_available() else 'cpu')

def get_sharepoint_context(site_url: str, client_id: str, client_secret: str):
    return ClientContext(site_url).with_credentials(
        ClientCredential(client_id, client_secret))
        
def update_progress(current_frame: int, total_frames: int, message: str):
    global current_process_id, ANALYSIS_ACTIVE
    if current_process_id and ANALYSIS_ACTIVE:
        progress = {
            "percent": min(100, (current_frame / total_frames) * 100),
            "message": message,
            "current_frame": current_frame,
            "total_frames": total_frames
        }
        PROGRESS_STORE[current_process_id] = progress

def detect_child_and_crop(frame):
    try:
        results = app.state.detection_model(frame, verbose=False)[0]
        class_ids = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        bboxes = results.boxes.xyxy.cpu().numpy()

        # Track best detections for each class
        child_bbox = None
        adult_bbox = None
        stranger_bbox = None
        adult_conf = 0
        stranger_conf = 0

        # Find highest confidence detections for each relevant class
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

        # Distance calculation functions
        def euclidean_distance(box1, box2):
            if box1 is None or box2 is None:
                return None
            center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
            center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
            return np.linalg.norm(np.array(center1) - np.array(center2))

        def calculate_diagonal(bbox):
            if bbox is None:
                return 0
            return np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)

        # Calculate adjusted distances with confidence check
        adjusted_distance_adult = None
        if adult_bbox is not None and adult_conf > 0.6:
            adult_diagonal = calculate_diagonal(adult_bbox)
            child_diagonal = calculate_diagonal(child_bbox)
            raw_dist = euclidean_distance(child_bbox, adult_bbox)
            if raw_dist is not None:
                adjusted_distance_adult = raw_dist * (abs(adult_diagonal - 1.67 * child_diagonal) / 1000)

        adjusted_distance_stranger = None
        if stranger_bbox is not None and stranger_conf > 0.6:
            stranger_diagonal = calculate_diagonal(stranger_bbox)
            child_diagonal = calculate_diagonal(child_bbox)
            raw_dist = euclidean_distance(child_bbox, stranger_bbox)
            if raw_dist is not None:
                adjusted_distance_stranger = raw_dist * (abs(stranger_diagonal - 1.67 * child_diagonal) / 1000)

        # Categorization logic with mutual inference
        def categorize(d):
            if d is None: return None
            return 2 if d < 20 else 1 if d < 200 else 0

        category_adult = categorize(adjusted_distance_adult)
        category_stranger = categorize(adjusted_distance_stranger)

        # Improved mutual inference logic
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

        # Count valid detections
        parent_count = np.sum((class_ids == 0) & (confidences > 0.6))
        stranger_count = np.sum((class_ids == 2) & (confidences > 0.6))

        # Crop child ROI
        x1, y1, x2, y2 = map(int, child_bbox)
        child_roi = frame[y1:y2, x1:x2]

        return child_roi, parent_count, category_adult, category_stranger

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
                np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                for key in LANDMARKS
                for (px, py), (cx, cy) in zip(prev_landmarks.get(key, []), current_landmarks.get(key, []))
            )
            valid_points = sum(len(landmarks) for landmarks in current_landmarks.values())
            if valid_points > 0:
                avg_diff = total_diff / valid_points
                movement_score = 2 if avg_diff > 6 else 1 if avg_diff > 3 else 0

        return movement_score, current_landmarks
    except Exception as e:
        logger.error(f"Facial processing error: {str(e)}")
        return 0, None

@app.post("/api/process-video")
async def process_video(
    video: UploadFile = File(None),
    file_id: str = Form(None),
    site_url: str = Form(None),
    client_id: str = Form(None),
    client_secret: str = Form(None),
    doc_library: str = Form(None),
    background_tasks: BackgroundTasks = None
):
    global current_process_id, ANALYSIS_ACTIVE
    current_process_id = str(uuid.uuid4())
    ANALYSIS_ACTIVE = True

    session_dir = os.path.join(OUTPUT_DIR, f"session_{current_process_id}")
    os.makedirs(session_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    cap = None
    video_path = None  # Initialize video_path

    try:
        # Handle file source
        if file_id:  # SharePoint file
            ctx = get_sharepoint_context(site_url, client_id, client_secret)
            file = ctx.web.get_file_by_id(file_id)
            ctx.load(file)
            ctx.execute_query()
            video_path = os.path.join(temp_dir, file.properties["Name"])
            with open(video_path, "wb") as f:
                file.download(f).execute_query()
        else:  # Direct upload
            video_path = os.path.join(temp_dir, video.filename)
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)

        # Video processing setup
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate and adjust FPS
        if original_fps <= 0 or original_fps > 240:
            original_fps = 30
            logger.warning(f"Using default FPS: {original_fps}")

        duration_seconds = int(total_frames / original_fps)
        total_processed = duration_seconds
        processed = 0

        logger.info(f"""
            Video Analysis Started
            ----------------------
            Path: {video_path}
            FPS: {original_fps}
            Total Frames: {total_frames}
            Duration: {duration_seconds}s
            Process Frames: {total_processed}
        """)

        results = []
        prev_landmarks = None
        prev_pose = None

        update_progress(0, total_processed, "Initializing analysis...")

        for second in range(duration_seconds):
            if not ANALYSIS_ACTIVE:
                break
                
            # Calculate actual frame number being processed
            current_frame = int(second * original_fps)
            update_progress(current_frame, total_frames, f"Processing frame {current_frame}/{total_frames}")

            # Get frame for current second
            target_frame = int(second * original_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Skipping second {second}")
                continue

            processed += 1
            timecode = f"{second//60:02d}:{second%60:02d}"
            
            try:
                # Child detection
                child_roi, _, adult_dist, stranger_dist = detect_child_and_crop(frame)
                if child_roi is None:
                    update_progress(processed, total_processed, f"{timecode}: No child detected")
                    continue

                # Face analysis
                face_score, curr_landmarks = facial_keypoints(child_roi, prev_landmarks)
                prev_landmarks = curr_landmarks

                # Pose analysis
                pose_kps = process_pose(child_roi)
                body_movement = 0.0
                if pose_kps is not None and prev_pose is not None:
                    valid_points = 0
                    total_movement = sum(
                        np.linalg.norm(curr - prev)
                        for prev, curr in zip(prev_pose, pose_kps)
                        if not (np.isnan(curr).any() or np.isnan(prev).any())
                    )
                    valid_points = len([p for p in pose_kps if not np.isnan(p).any()])
                    if valid_points > 0:
                        body_movement = round(total_movement / valid_points, 2)
                prev_pose = pose_kps

                # Store results
                results.append({
                    "second": second,
                    "timecode": timecode,
                    "face_movement": face_score,
                    "body_movement": body_movement,
                    "adult_distance": adult_dist if adult_dist is not None else -1,
                    "stranger_distance": stranger_dist if stranger_dist is not None else -1
                })

                update_progress(
                    processed,
                    total_processed,
                    f"{timecode}: Child detected" + 
                    ("" if adult_dist is None else f" | Adult: {adult_dist}") +
                    ("" if stranger_dist is None else f" | Stranger: {stranger_dist}")
                )

            except Exception as e:
                logger.error(f"Second {second} error: {str(e)}")
                update_progress(processed, total_processed, f"{timecode}: Error processing")

        # Save results
        df = pd.DataFrame(results)
        csv_path = os.path.join(session_dir, "analysis.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Analysis complete. Generated {len(results)} records")

        return FileResponse(csv_path, filename="analysis.csv")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(500, str(e))
    finally:
        ANALYSIS_ACTIVE = False
        if cap:
            cap.release()
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/api/delete-video")
async def delete_video(process_id: str = Form(...)):
    if process_id in UPLOADED_VIDEOS:
        try:
            video_info = UPLOADED_VIDEOS.pop(process_id)
            paths = [
                video_info["video_path"],
                video_info["temp_dir"],
                video_info["session_dir"]
            ]
            for path in paths:
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path, ignore_errors=True)
            return {"status": "deleted"}
        except Exception as e:
            logger.error(f"Deletion error: {str(e)}")
            raise HTTPException(500, "Deletion failed")
    raise HTTPException(404, "Video not found")

@app.get("/api/progress")
async def progress_stream():
    async def event_generator():
        last_update = None
        while True:
            if current_process_id in PROGRESS_STORE:
                current = PROGRESS_STORE[current_process_id]
                if current != last_update:
                    last_update = current
                    yield f"data: {json.dumps(current)}\n\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/cancel-analysis")
async def cancel_analysis():
    global ANALYSIS_ACTIVE, current_process_id
    ANALYSIS_ACTIVE = False
    if current_process_id and current_process_id in PROGRESS_STORE:
        PROGRESS_STORE[current_process_id] = {
            "percent": 0,
            "message": "Analysis cancelled by user",
            "current_frame": 0,
            "total_frames": 0
        }
        current_process_id = None  # Reset process ID
    return {"status": "cancelled"}

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # Skip API routes and static files
    if full_path.startswith(("api/", "static/")):
        raise HTTPException(status_code=404)
    
    # Serve index.html for all other routes
    frontend_path = Path("frontend/index.html")  # Relative to where you run the app
    if not frontend_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Frontend not found at {frontend_path.absolute()}"
        )
    return FileResponse(frontend_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
