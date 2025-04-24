import os
import cv2
import uuid
import json
import time
import re
import subprocess
import asyncio
import logging
import numpy as np
import pandas as pd
import tempfile
import warnings
import shutil
from pathlib import Path
from PIL import Image
import ffmpeg
import torch
import torchvision.transforms as T
from ultralytics import YOLO
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.midas_utils.transforms import Compose, Resize, NormalizeImage, PrepareForNet

################################################# 
# Initialize application
#################################################
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


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve frontend files
static_dir = Path(__file__).parent.parent / "frontend" / "static" 
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configuration
MODEL_PATH = "parents_child_stranger.pt"
MAX_VIDEO_SIZE = 500 * 1024 * 1024
OUTPUT_DIR = "analysis_output"
UPLOADED_VIDEOS = {}  # Track uploaded video session
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global state
PROGRESS_STORE = {}
ANALYSIS_ACTIVE = False

@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )

@app.on_event("startup")
async def initialize_models():
    """Initialize models with warmup inference"""
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing models on {device}")

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

def update_progress(process_id: str, current: int, total: int, message: str):
    """Update progress store with analysis status"""
    PROGRESS_STORE[process_id] = {
        "percent": min(100, (current / total) * 100),
        "message": message,
        "current": current,
        "total": total,
        "status": "processing"
    }

################################################# 
# Initialize Models
#################################################

# Child detection and image cropping
def detect_child_and_crop(frame):
    try:
        results = app.state.detection_model.predict(frame, verbose=False)[0]
        class_ids = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        bboxes = results.boxes.xyxy.cpu().numpy()
        child_bbox = None
        
        for box, cls, conf in zip(bboxes, class_ids, confidences):
            if conf > 0.6:
                if cls == 1:
                    child_bbox = box
                elif cls == 0:
                    adult_bbox = box
                elif cls == 2:
                    stranger_bbox = box

        if child_bbox is None:
            return None

        x1, y1, x2, y2 = map(int, child_bbox)
        # Validate and clamp coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x1 >= x2 or y1 >= y2:
            logger.warning("Invalid child bounding box")
            return None
        
        child_roi = frame[y1:y2, x1:x2]
        if child_roi.size == 0:
            logger.warning("Empty child ROI")
            return None

        return child_roi
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return None

def load_depth_model():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = torch.hub.load(
                'intel-isl/MiDaS', 
                'MiDaS_small',
                pretrained=True,
                trust_repo=True
            ).float()
        model.eval().to(device)
        print("Successfully loaded MiDaS model from torch.hub")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load MiDaS model: {e}")

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
Resize = midas_transforms.Resize
NormalizeImage = midas_transforms.NormalizeImage
PrepareForNet = midas_transforms.PrepareForNet

# Define transform pipeline
transform_pipeline = T.Compose([
    lambda img: {"image": np.array(img.convert("RGB"), dtype=np.float32) / 255.0},
    Resize(
        256, 256, resize_target=None, keep_aspect_ratio=True,
        ensure_multiple_of=32, resize_method="upper_bound",
        image_interpolation_method=cv2.INTER_CUBIC
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
    lambda sample: torch.from_numpy(sample["image"]),
])

# Load model once
depth_model = load_depth_model()

def calculate_distance_between_objects(frame, obj1_label, obj2_label):
    results = app.state.detection_model.predict(frame, verbose=False)[0]
    labels = results.names if hasattr(results, 'names') else {}

    obj1_center = None
    obj2_center = None

    for box in results.boxes:
        cls = int(box.cls[0].item())
        label = labels.get(cls, str(cls))

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if label.lower() == obj1_label.lower():
            obj1_center = center
        elif label.lower() == obj2_label.lower():
            obj2_center = center

    # Validation checks with proper error handling
    if obj1_center is None:
        print(f"Important warning: {obj1_label} not detected.")
        return None
        
    if obj2_center is None:
        if obj2_label.lower() != "stranger":
            print(f"Warning: {obj2_label} not detected.")
        return None

    # Add coordinate validation
    def validate_coord(coord):
        return isinstance(coord, tuple) and len(coord) == 2 and \
               all(isinstance(v, (int, float)) for v in coord)

    if not validate_coord(obj1_center) or not validate_coord(obj2_center):
        print("Invalid coordinates detected")
        return None

    try:
        # Estimate depth
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = transform_pipeline(img_pil)

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = depth_model(input_tensor)
            depth_map = output.squeeze().cpu().numpy()

        # Rescale object centers with safety checks
        original_h, original_w = frame.shape[:2]
        depth_h, depth_w = depth_map.shape

        def safe_scale(coord, orig_dim, target_dim):
            try:
                return int((coord / orig_dim) * target_dim)
            except ZeroDivisionError:
                return 0

        # Corrected scaling calls
        x1 = safe_scale(obj1_center[0], original_w, depth_w)
        y1 = safe_scale(obj1_center[1], original_h, depth_h)
        x2 = safe_scale(obj2_center[0], original_w, depth_w)
        y2 = safe_scale(obj2_center[1], original_h, depth_h)
        
        # Depth calculation with bounds checking
        def get_depth(x, y):
            x = max(0, min(depth_w-1, x))
            y = max(0, min(depth_h-1, y))
            return depth_map[y, x]

        d1 = get_depth(x1, y1)
        d2 = get_depth(x2, y2)

        if d1 <= 0 or d2 <= 0:
            return None

        # 3D coordinate conversion
        fx = fy = 500  # Focal length assumption
        cx, cy = depth_w // 2, depth_h // 2

        point1 = (
            (x1 - cx) * d1 / fx,
            (y1 - cy) * d1 / fy,
            d1
        )
        point2 = (
            (x2 - cx) * d2 / fx,
            (y2 - cy) * d2 / fy,
            d2
        )

        return float(np.linalg.norm(np.array(point1) - np.array(point2)))

    except Exception as e:
        logger.error(f"Distance calculation error: {str(e)}")
        return None

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

def facial_keypoints(image, prev_landmarks=None):
    if image is None:
        logger.error("Received None frame")
        return 0, None
    try:
        h, w = image.shape[:2]
    except AttributeError:
        logger.error("Invalid image type")
        return 0, None
    if h == 0 or w == 0 or image.size == 0:
        logger.error("Received empty frame")
        return 0, None

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

def process_pose(image):
    if image is None:
        return None
    try:
        results = app.state.pose_model(image, verbose=False)
        if results and hasattr(results[0], 'keypoints'):
            return results[0].keypoints.xy[0].cpu().numpy()
        return None
    except Exception as e:
        logger.error(f"Pose processing error: {str(e)}")
        return None

def calculate_body_movement(current_pose, previous_pose):
    if current_pose is None or previous_pose is None:
        return 0.0
    
    valid_points = 0
    total_movement = 0.0
    
    for prev, curr in zip(previous_pose, current_pose):
        if not (np.isnan(prev).any() or np.isnan(curr).any()):
            valid_points += 1
            total_movement += abs(np.linalg.norm(curr - prev))
    
    return total_movement

#################################################
# Preparing for Video Processing
#################################################

def time_to_seconds(timestamp):
    return sum(x * int(t) for x, t in zip([3600, 60, 1], timestamp.split(':')))

def format_progress_message(stage, current, total, extras=None):
    base = f"{stage} - Frame {current}/{total}"
    if extras:
        return f"{base} - {', '.join(f'{k}: {v}' for k,v in extras.items())}"
    return base

def crop_video(video_path: str, timestamp1: str, timestamp2: str, timestamp3: str, temp_dir: str, ffmpeg_path: str = 'ffmpeg'):
    """
    Crop the video into two clips:
    - First clip: timestamp1 to timestamp2
    - Second clip: timestamp2 to timestamp3
    """
    import os, uuid, subprocess
    ts1 = time_to_seconds(timestamp1)
    ts2 = time_to_seconds(timestamp2)
    ts3 = time_to_seconds(timestamp3)

    first_clip_path = os.path.join(temp_dir, f"clip1_{uuid.uuid4()}.mp4")
    second_clip_path = os.path.join(temp_dir, f"clip2_{uuid.uuid4()}.mp4")

    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg binary not found at {ffmpeg_path}")

    # Clip 1
    dur1 = ts2 - ts1
    command1 = [
        ffmpeg_path, '-y', '-i', video_path,
        '-ss', str(ts1), '-t', str(dur1),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', first_clip_path
    ]

    # Clip 2
    dur2 = ts3 - ts2
    command2 = [
        ffmpeg_path, '-y', '-i', video_path,
        '-ss', str(ts2), '-t', str(dur2),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', second_clip_path
    ]

    logger.info("Running command: %s", ' '.join(command1))
    subprocess.run(command1, check=True)
    logger.info("Running command: %s", ' '.join(command2))
    subprocess.run(command2, check=True)

    return first_clip_path, second_clip_path

#################################################
# Video Processing Loop
#################################################

async def process_video_async(process_id: str, video_path: Path, session_dir: Path,
                              timestamp1: str, timestamp2: str, timestamp3: str, temp_dir: Path,
                              ffmpeg_path: str = "ffmpeg"):
    try:
        # Initialize progress tracking
        PROGRESS_STORE[process_id] = {
            "status": "processing",
            "percent": 0,
            "message": "Initializing",
            "result": None,
            "error": None
        }

        # Validate timestamps
        def validate_timestamp(t):
            parts = t.split(':')
            return (len(parts) == 3 and all(p.isdigit() for p in parts))
        
        if not all(validate_timestamp(ts) for ts in [timestamp1, timestamp2, timestamp3]):
            raise ValueError("Invalid timestamp format")

        # Crop video
        PROGRESS_STORE[process_id].update({
            "message": "Cropping video segments",
            "percent": 5
        })
        
        freeplay_video, experiment_video = await asyncio.to_thread(
            crop_video,
            str(video_path),
            timestamp1,
            timestamp2,
            timestamp3,
            str(temp_dir)
        )

        # Process freeplay segment
        PROGRESS_STORE[process_id].update({
            "message": "Analyzing freeplay movement",
            "percent": 10
        })
        
        cap_fp = cv2.VideoCapture(str(freeplay_video))
        fps_fp = cap_fp.get(cv2.CAP_PROP_FPS)
        freeplay_movements = []
        prev_pose = None

        for sec in range(int(cap_fp.get(cv2.CAP_PROP_FRAME_COUNT) / fps_fp)):
            await asyncio.sleep(0)
            print(f"Processing freeplay frame {sec}")
            if PROGRESS_STORE[process_id]["status"] == "cancelled":
                break
                
            cap_fp.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps_fp))
            ret, frame = cap_fp.read()
    
            if not ret or frame is None or frame.size == 0:
                logger.warning("Skipping invalid frame")
                continue

            child_roi = detect_child_and_crop(frame)
            pose_kps = process_pose(child_roi)
            if pose_kps is not None:
                mv = calculate_body_movement(pose_kps, prev_pose)
                freeplay_movements.append(mv)
                prev_pose = pose_kps

        cap_fp.release()
        freeplay_movement = np.mean(freeplay_movements) if freeplay_movements else 0.0

        # Process experiment segment
        PROGRESS_STORE[process_id].update({
            "message": "Analyzing experiment",
            "percent": 30
        })
        
        cap_exp = cv2.VideoCapture(str(experiment_video))
        fps_exp = cap_exp.get(cv2.CAP_PROP_FPS)
        results = []
        prev_landmarks = None
        prev_pose = None
        total_seconds = int(cap_exp.get(cv2.CAP_PROP_FRAME_COUNT) / fps_exp)
        for sec in range(total_seconds):
            progress_percent = int((sec / total_seconds) * 65) + 30  # 30-95% range
            PROGRESS_STORE[process_id].update({
                "percent": progress_percent,
                "current": sec,
                "total": total_seconds
            })
            
            print(f"Processing experiment frame {sec}")
            if PROGRESS_STORE[process_id]["status"] == "cancelled":
                break

            ret, frame = cap_exp.read()
            
            if not ret or frame is None or frame.size == 0:
                logger.warning("Skipping invalid frame")
                continue

            # Update progress
            progress = int(sec/int(cap_exp.get(cv2.CAP_PROP_FRAME_COUNT) / fps_exp))
            PROGRESS_STORE[process_id].update({
                "percent": progress,
                "current": sec,
                "total": total_frames
            })

            # Analysis logic
            child_roi = detect_child_and_crop(frame)
            if child_roi is None:
                continue

            face_score, curr_landmarks = facial_keypoints(child_roi, prev_landmarks)
            pose_kps = process_pose(child_roi)
            body_mv = calculate_body_movement(pose_kps, prev_pose)
            mov_ratio = body_mv / freeplay_movement if freeplay_movement else 0.0
            
            parent_dist = calculate_distance_between_objects(frame, "Child", "Adult")
            stranger_dist = calculate_distance_between_objects(frame, "Child", "Stranger")

            results.append({
                "second": sec,
                "parent_dist": parent_dist,
                "stranger_dist": stranger_dist,
                "face_movement": face_score,
                "body_movement": mov_ratio
            })

            prev_landmarks = curr_landmarks
            prev_pose = pose_kps

        # Save final results
        csv_path = session_dir / "analysis.csv"
        pd.DataFrame(results).to_csv(csv_path, index=False)

        # Add validation
        if not csv_path.exists():
            raise FileNotFoundError(f"Analysis results not created at {csv_path}")
        if csv_path.stat().st_size == 0:
            csv_path.unlink()
            raise RuntimeError("Empty analysis results file created")

        # Update progress after successful CSV creation
        PROGRESS_STORE[process_id].update({
            "status": "completed",
            "percent": 100,
            "message": "Analysis complete",
            "result": str(csv_path)
        })

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        PROGRESS_STORE[process_id].update({
            "status": "error",
            "error": str(e),
            "percent": 100
        })
        
    finally:
        cleanup_done = False
        resources_to_clean = []
        
        try:
            # Track cleanup resources
            resources_to_clean = []
            
            # Release video captures
            if 'cap_fp' in locals():
                resources_to_clean.append(("Freeplay video capture", cap_fp))
            if 'cap_exp' in locals():
                resources_to_clean.append(("Experiment video capture", cap_exp))
            
            # Clean up video captures
            for name, cap in resources_to_clean:
                try:
                    if cap.isOpened():
                        cap.release()
                        logger.info(f"Released {name}")
                except Exception as e:
                    logger.error(f"Failed to release {name}: {str(e)}")
            
            # Clean temp directory
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned temp directory: {temp_dir}")
                except Exception as e:
                    logger.error(f"Failed to clean temp directory: {str(e)}")
                    # Try alternative cleanup
                    try:
                        subprocess.run(['rm', '-rf', str(temp_dir)], check=True)
                    except:
                        logger.critical("Complete directory cleanup failure!")

            cleanup_done = True
            
        except Exception as cleanup_err:
            logger.error(f"Cleanup error: {cleanup_err}")
            cleanup_done = False
            
        finally:
            if not cleanup_done:
                logger.critical("""
                    CRITICAL CLEANUP FAILURE!
                    Potential resource leaks detected:
                    Unreleased resources: %s
                    Temp directory exists: %s (%s)
                    """, 
                    [name for name, _ in resources_to_clean],
                    temp_dir.exists(),
                    temp_dir if temp_dir.exists() else ""
                )
                
            # Final verification
            remaining = [
                (name, cap) for name, cap in resources_to_clean 
                if cap.isOpened()
            ]
            if remaining:
                logger.critical(
                    "Unreleased resources remaining: %s", 
                    [name for name, _ in remaining]
                )
            


#################################################
# API Endpoints
#################################################

@app.post("/api/process-video")
async def start_processing(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    timestamp1: str = Form(...),
    timestamp2: str = Form(...),
    timestamp3: str = Form(...)
):
    # Validate timestamps
    for ts in (timestamp1, timestamp2, timestamp3):
        if not re.match(r"^\d{2}:\d{2}:\d{2}$", ts):
            raise HTTPException(400, "Invalid timestamp format (HH:MM:SS)")

    process_id = str(uuid.uuid4())
    session_dir = Path(OUTPUT_DIR).absolute() / f"session_{process_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for processing
    temp_dir = Path(tempfile.mkdtemp())

    # Save video to session directory
    video_path = session_dir / video.filename
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Update background task to include temp_dir
    background_tasks.add_task(
        process_video_async,
        process_id,
        video_path,
        session_dir,
        timestamp1,
        timestamp2,
        timestamp3,
        temp_dir,  # Pass the temp_dir here
    )

    return {"process_id": process_id}

@app.get("/api/progress/{process_id}")
async def progress_stream(process_id: str):
    async def event_generator():
        last = None
        while True:
            if process_id in PROGRESS_STORE:
                current = PROGRESS_STORE[process_id]
                if current != last:
                    last = current
                    yield f"data: {json.dumps(current)}\n\n"
                if current["status"] in ["completed", "error", "cancelled"]:
                    break
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/api/results/{process_id}")
async def results(process_id: str):
    if process_id not in PROGRESS_STORE:
        raise HTTPException(404, detail="Process ID not found")
    
    status = PROGRESS_STORE[process_id]
    
    if status["status"] == "completed":
        csv_path = Path(status["result"])
        try:
            return FileResponse(
                csv_path,
                media_type="text/csv",
                filename="analysis_results.csv",
                headers={"X-Analysis-Complete": "true"}
            )
        except FileNotFoundError:
            logger.error(f"Missing results file: {csv_path}")
            PROGRESS_STORE[process_id].update({
                "status": "error",
                "error": "Results file missing"
            })
            raise HTTPException(500, "Results file missing")
    
    raise HTTPException(425, detail="Analysis not complete yet")

@app.post("/api/cancel-analysis")
async def cancel_analysis(process_id: str = Form(...)):
    if process_id in PROGRESS_STORE:
        PROGRESS_STORE[process_id].update({"status": "cancelled", "message": "Cancelled by user"})
    return {"status": "cancelled"}

@app.post("/api/delete-video")
async def delete_video(process_id: str = Form(...)):
    if process_id in PROGRESS_STORE:
        PROGRESS_STORE.pop(process_id, None)
        return {"status": "deleted"}
    raise HTTPException(404, detail="Video not found")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    if full_path.startswith(("api/", "static/")):
        raise HTTPException(status_code=404)
    frontend = Path("frontend/index.html")
    if not frontend.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(frontend)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


