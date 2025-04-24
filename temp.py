import os
import cv2
import uuid
import json
import time
import subprocess
import asyncio
import logging
import numpy as np
import pandas as pd
import tempfile
import warnings
import shutil
from pathlib import Path
from math import ceil
import torch
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
current_process_id = None
ANALYSIS_ACTIVE = False

# Add these under your configuration section
DEPTH_MIN_CONFIDENCE = 0.5
PIXEL_DISTANCE_THRESHOLD = 20
FOCAL_LENGTH = 1109  # Adjust based on your camera

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
        device = torch.device('cpu')
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
    try:
        current_num = int(current)
        total_num = int(total)
        PROGRESS_STORE[process_id] = {
            "percent": min(100, (current_num / total_num) * 100),
            "message": str(message),
            "current": current_num,
            "total": total_num,
            "status": "processing"
        }
    except (TypeError, ValueError) as e:
        logger.error(f"Progress update error: {str(e)}")
        PROGRESS_STORE[process_id] = {
            "percent": 0,
            "message": "Error calculating progress",
            "current": 0,
            "total": 1,
            "status": "error"
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
    # Try multiple loading methods in sequence
    try:
        # Method 1: Try torch.hub with all dependencies
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = torch.hub.load(
                'intel-isl/MiDaS', 
                'MiDaS_small',
                pretrained=True,
                trust_repo=True  # Required for newer PyTorch versions
            ).float()
        model.eval()
        print("Successfully loaded model via torch.hub")
        return model
        
    except Exception as hub_error:
        print(f"Hub loading failed: {hub_error}")
        
        # Method 2: Try local JIT load
        model_path = Path(__file__).parent / "midas_utils" / "model.pt"
        if model_path.exists():
            try:
                model = torch.jit.load(str(model_path), map_location='cpu')
                model.eval()
                print("Successfully loaded local JIT model")
                return model
            except Exception as jit_error:
                print(f"JIT load failed: {jit_error}")
        
        # Method 3: Download fresh copy
        try:
            import requests
            from tqdm import tqdm
            
            url = "https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
            temp_path = model_path.with_name("temp_model.pt")
            
            print("Downloading fresh model copy...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save with progress bar
            total_size = int(response.headers.get('content-length', 0))
            with open(temp_path, 'wb') as f, tqdm(
                desc=str(temp_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
            
            # Verify download
            if temp_path.stat().st_size < 80 * 1024 * 1024:  # ~80MB
                raise ValueError("Downloaded file too small")
            
            model = torch.jit.load(str(temp_path))
            model.eval()
            
            # Replace old model file
            if model_path.exists():
                model_path.unlink()
            temp_path.rename(model_path)
            
            print("Successfully downloaded and loaded fresh model")
            return model
            
        except Exception as download_error:
            print(f"All loading methods failed: {download_error}")
            raise RuntimeError("Could not load depth estimation model")

try:
    depth_model = load_depth_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_model = depth_model.to(device)
except Exception as e:
    print(f"Critical error loading depth model: {e}")

# Custom transformation pipeline: note the final lambda does NOT unsqueeze here.
transform_pipeline = Compose([
    Resize(256),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet()
])

def calculate_distance_between_objects(frame, obj1_label, obj2_label, depth_map=None):
    try:
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame")
            return None

        # Use predict() for detection model
        results = app.state.detection_model.predict(frame, verbose=False)[0]
        class_ids = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        bboxes = results.boxes.xyxy.cpu().numpy()

        obj1_det = {'conf': -1, 'box': None}
        obj2_det = {'conf': -1, 'box': None}

        for box, cls_idx, conf in zip(bboxes, class_ids, confidences):
            if conf < DEPTH_MIN_CONFIDENCE:
                continue

            class_name = app.state.detection_model.names[int(cls_idx)]
            if class_name.lower() == obj1_label.lower():
                if conf > obj1_det['conf']:
                    obj1_det = {'conf': conf, 'box': box}
            elif class_name.lower() == obj2_label.lower():
                if conf > obj2_det['conf']:
                    obj2_det = {'conf': conf, 'box': box}

        if obj1_det['box'] is None or obj2_det['box'] is None:
            return None

        def validate_box(box):
            return (
                isinstance(box, np.ndarray) and 
                box.shape == (4,) and 
                (box[2] > box[0]) and 
                (box[3] > box[1])
            )

        if not all(validate_box(b) for b in [obj1_det['box'], obj2_det['box']]):
            return None

        def get_center(box, frame_shape):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cx = np.clip(cx, 0, frame_shape[1]-1)
            cy = np.clip(cy, 0, frame_shape[0]-1)
            return (int(cx), int(cy))

        obj1_center = get_center(obj1_det['box'], frame.shape)
        obj2_center = get_center(obj2_det['box'], frame.shape)

        pixel_dist = np.linalg.norm(np.array(obj1_center) - np.array(obj2_center))
        if pixel_dist < PIXEL_DISTANCE_THRESHOLD:
            return {
                "object_1": obj1_label,
                "object_2": obj2_label,
                "distance_3d": 0.0
            }

        try:
            # Only compute depth map if not provided
            if depth_map is None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform_pipeline(img_rgb).to(device)
                
                with torch.no_grad(), torch.amp.autocast(device_type=device_type):
                    depth_map = depth_model(input_tensor)
                    if depth_map.dtype == torch.bfloat16:
                        depth_map = depth_map.to(torch.float32)
                    depth_map = depth_map.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(f"Depth processing failed: {str(e)}")
            return None

        def get_depth(center, depth_map):
            x, y = map(int, np.round(center))
            h, w = depth_map.shape
            patch = depth_map[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)]
            if patch.size == 0:
                return 0.0
            masked_patch = np.ma.masked_equal(patch, 0.0)
            return masked_patch.mean() if masked_patch.count() > 0 else 0.0

        depth1 = get_depth(obj1_center, depth_map)
        depth2 = get_depth(obj2_center, depth_map)

        if depth1 <= 0 or depth2 <= 0:
            return None

        def pixel_to_3d(x, y, depth):
            fx = fy = FOCAL_LENGTH
            cx = frame.shape[1] // 2
            cy = frame.shape[0] // 2
            x_3d = (x - cx) * depth / fx
            y_3d = (y - cy) * depth / fy
            return (x_3d, y_3d, depth)

        point1 = pixel_to_3d(*obj1_center, depth1)
        point2 = pixel_to_3d(*obj2_center, depth2)

        distance_3d = float(np.linalg.norm(np.array(point1) - np.array(point2)))

        return {
            "object_1": obj1_label,
            "object_2": obj2_label,
            "distance_3d": distance_3d
        }

    except Exception as e:
        logger.error(f"Distance calculation error: {str(e)}", exc_info=True)
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
    ts1 = time_to_seconds(timestamp1)
    ts2 = time_to_seconds(timestamp2)
    ts3 = time_to_seconds(timestamp3)

    first_clip_path = os.path.join(temp_dir, f"clip1_{uuid.uuid4()}.mp4")
    second_clip_path = os.path.join(temp_dir, f"clip2_{uuid.uuid4()}.mp4")

    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"ffmpeg binary not found at {ffmpeg_path}")

    # Use -i, then -ss and -t for reliable clip lengths
    dur1 = ts2 - ts1
    command1 = [
        ffmpeg_path, '-y',
        '-i', video_path,
        '-ss', str(ts1),
        '-t', str(dur1),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', first_clip_path
    ]

    dur2 = ts3 - ts2
    command2 = [
        ffmpeg_path, '-y',
        '-i', video_path,
        '-ss', str(ts2),
        '-t', str(dur2),
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
    global current_process_id
    current_process_id = process_id

    # 1) Seed progress before any work
    PROGRESS_STORE[process_id] = {
        "percent": 0,
        "message": "Cropping videos",
        "current": 0,
        "total": None,
        "status": "processing"
    }

    # 2) Validate timestamps
    def validate_timestamp(t):
        parts = t.split(':')
        return (len(parts) == 3 and all(p.isdigit() for p in parts)
                and 0 <= int(parts[0]) < 24 and 0 <= int(parts[1]) < 60 and 0 <= int(parts[2]) < 60)

    if not all(validate_timestamp(ts) for ts in [timestamp1, timestamp2, timestamp3]):
        PROGRESS_STORE[process_id].update({"status": "error", "error": "Invalid timestamp format"})
        return

    # 3) Crop video into two clips
    try:
        freeplay_video, experiment_video = crop_video(str(video_path), timestamp1, timestamp2, timestamp3,
                                                      str(temp_dir), ffmpeg_path)
    except Exception as e:
        logger.error("Cropping error: %s", e)
        PROGRESS_STORE[process_id].update({"status": "error", "error": str(e)})
        return

    # 4) Process freeplay segment
    try:
        PROGRESS_STORE[process_id].update({"message": "Processing freeplay"})
        cap_fp = cv2.VideoCapture(str(freeplay_video))
        if not cap_fp.isOpened(): raise RuntimeError(f"Failed to open freeplay video")

        fps_fp = cap_fp.get(cv2.CAP_PROP_FPS)
        frames_fp = int(cap_fp.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_fp = int(frames_fp / fps_fp)
        freeplay_movements, prev_pose = [], None

        for sec in range(dur_fp):
            print(f"Processing freeplay frame {sec}")
            if PROGRESS_STORE[process_id]["status"] == "cancelled": break
            cap_fp.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps_fp))
            ret, frame = cap_fp.read()
            if not ret: continue
            try:
                child_roi = detect_child_and_crop(frame)
                pose_kps = process_pose(child_roi)
                mv = calculate_body_movement(pose_kps, prev_pose)
                prev_pose, freeplay_movements = pose_kps, freeplay_movements + [mv]
            except Exception as e:
                logger.error("Freeplay error at %d: %s", sec, e)
        cap_fp.release()
        freeplay_movement = float(np.mean(freeplay_movements)) if freeplay_movements else 0.0
    except Exception as e:
        logger.error("Freeplay processing failed: %s", e)
        PROGRESS_STORE[process_id].update({"status": "error", "error": str(e)})
        return

    # 5) Process experiment segment
    try:
        PROGRESS_STORE[process_id].update({"message": "Analyzing experiment"})
        cap_exp = cv2.VideoCapture(str(experiment_video))
        if not cap_exp.isOpened(): raise RuntimeError(f"Failed to open experiment video")

        fps_exp = cap_exp.get(cv2.CAP_PROP_FPS)
        frames_exp = int(cap_exp.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_exp = int(frames_exp / fps_exp)
        PROGRESS_STORE[process_id].update({"total": dur_exp})

        results, prev_landmarks, prev_pose = [], None, None
        for sec in range(dur_exp):
            print(f"Processing experiment frame {sec}")
            if PROGRESS_STORE[process_id]["status"] == "cancelled": break
            cap_exp.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps_exp))
            ret, frame = cap_exp.read()
            if not ret: continue

            if frame is None:
                logger.warning(f"Received None frame at second {sec+1}")
                results.append({
                    "second": sec+1,
                    "parent_dist": None,
                    "stranger_dist": None,
                    "face_movement": None,
                    "body_movement": None
                })
                continue

            PROGRESS_STORE[process_id].update({
                "current": sec+1,
                "percent": int((sec+1)/dur_exp*100)
            })

            try:
                # Depth map calculation
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform_pipeline(img_rgb).to(device)
                with torch.no_grad():
                    depth_map = depth_model(input_tensor).squeeze().cpu().numpy()
                
                # Analysis
                child_roi = detect_child_and_crop(frame)
                face_score, curr_landmarks = facial_keypoints(child_roi, prev_landmarks)
                pose_kps = process_pose(child_roi)
                body_mv = calculate_body_movement(pose_kps, prev_pose)
                mov_ratio = body_mv/freeplay_movement if freeplay_movement else 0.0
                prev_landmarks = curr_landmarks
                prev_pose = pose_kps
                
                # Distance calculations
                parent_dist = calculate_distance_between_objects(frame, "Child", "Adult", depth_map)
                stranger_dist = calculate_distance_between_objects(frame, "Child", "Stranger", depth_map)

                results.append({
                    "second": sec+1,
                    "parent_dist": parent_dist,
                    "stranger_dist": stranger_dist,
                    "face_movement": face_score,
                    "body_movement": mov_ratio
                })

            except Exception as e:
                logger.error("Experiment error at %d: %s", sec, e)
            if results:
                pd.DataFrame(results).to_csv(session_dir/"analysis.csv", index=False)
        cap_exp.release()
        if results:
            PROGRESS_STORE[process_id].update({"status": "completed", "result": str(session_dir/"analysis.csv")})
        else:
            raise RuntimeError("No experiment results")
    except Exception as e:
        logger.error("Experiment processing error: %s", e)
        PROGRESS_STORE[process_id].update({"status": "error", "error": str(e)})
    finally:
        try:
            if video_path.exists(): video_path.unlink()
            shutil.rmtree(str(temp_dir), ignore_errors=True)
        except: pass

#################################################
# API Endpoints
#################################################

@app.post("/api/process-video") 
async def start_processing(
    video: UploadFile = File(...),
    timestamp1: str = Form(...),
    timestamp2: str = Form(...),
    timestamp3: str = Form(...)
):
    """
    Synchronously process the video and return the resulting CSV.
    Also updates PROGRESS_STORE for SSE progress.
    """
    # Validate timestamp format
    for ts in (timestamp1, timestamp2, timestamp3):
        parts = ts.split(':')
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise HTTPException(400, detail="Invalid timestamp format (HH:MM:SS)")

    # Prepare IDs and directories
    process_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.mkdtemp())
    session_dir = Path(OUTPUT_DIR) / f"session_{process_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    upload_dir = Path(tempfile.mkdtemp(prefix=f"upload_{process_id}_"))
    video_path = upload_dir / video.filename
    with open(video_path, 'wb') as f:
        f.write(await video.read())

    try:
        # Run processing directly
        await process_video_async(
            process_id,
            video_path,
            session_dir,
            timestamp1,
            timestamp2,
            timestamp3,
            temp_dir
        )

        csv_file = session_dir / "analysis.csv"
        if not csv_file.exists():
            raise HTTPException(500, detail="Analysis completed but no CSV found")

        # Return the CSV file once done
        return FileResponse(
            path=str(csv_file),
            media_type="text/csv",
            filename=csv_file.name
        )

    finally:
        # Cleanup upload_dir only; keep session_dir for download
        try:
            shutil.rmtree(str(upload_dir), ignore_errors=True)
        except:
            pass

@app.get("/api/progress")
async def progress_stream_root():
    """
    SSE endpoint without process_id uses the most recent process_id
    """
    async def event_generator():
        last = None
        while True:
            if current_process_id and current_process_id in PROGRESS_STORE:
                current = PROGRESS_STORE[current_process_id]
                if current != last:
                    last = current
                    yield f"data: {json.dumps(current)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )

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
            await asyncio.sleep(0.5)
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )

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


