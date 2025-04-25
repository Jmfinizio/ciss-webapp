import datetime
import os
import cv2
import uuid
import json
import time
import re
import subprocess
import uuid
import asyncio
import joblib
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
MODEL_PATH = "yolo_retrained_model.pt"
MAX_VIDEO_SIZE = 500 * 1024 * 1024
OUTPUT_DIR = Path("analysis_output")
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
        img_pil = Image.fromarray(img_rgb)  # Convert to PIL Image first
        input_tensor = transform_pipeline(img_pil).to(device)

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
        fx = fy = 1109  # Focal length assumption
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

def process_freeplay(process_id: str, freeplay_video: str) -> float:
    """
    Sample one frame per second from the freeplay clip,
    compute body‐movement metrics and return the average.
    """
    PROGRESS_STORE[process_id].update({"message": "Processing freeplay"})
    cap = cv2.VideoCapture(freeplay_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open freeplay video at {freeplay_video}")

    # Determine clip duration in seconds
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total_frames / fps

    movements = []
    prev_pose = None

    for sec in range(int(duration)):
        print(f"Processing freeplay frame {sec}")
        if PROGRESS_STORE[process_id]["status"] == "cancelled":
            break

        # Seek by time (ms)
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.warning(f"Freeplay: no frame at {sec}s")
            continue

        PROGRESS_STORE[process_id].update({
            "current": sec,
            "percent": 10 + int((sec + 1) / duration * 30)
        })

        try:
            child_roi = detect_child_and_crop(frame)
            pose_kps = process_pose(child_roi)
            mv = calculate_body_movement(pose_kps, prev_pose)
            movements.append(mv)
            prev_pose = pose_kps
        except Exception as e:
            logger.error(f"Freeplay error at {sec}s: {e}", exc_info=True)

    cap.release()
    return float(np.mean(movements)) if movements else 0.0

def process_experiment(process_id: str, experiment_video: str, freeplay_movement: float) -> pd.DataFrame:
    """
    Sample one frame per second from the experiment clip,
    compute all metrics, and return a DataFrame.
    """
    PROGRESS_STORE[process_id].update({"message": "Analyzing experiment"})
    cap = cv2.VideoCapture(experiment_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open experiment video at {experiment_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total_frames / fps
    PROGRESS_STORE[process_id].update({"total": int(duration)})

    results = []
    prev_landmarks = None
    prev_pose = None

    for sec in range(int(duration)):
        print(f"Processing experiment frame {sec}")
        if PROGRESS_STORE[process_id]["status"] == "cancelled":
            break

        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.warning(f"Experiment: no frame at {sec}s")
            results.append({
                "second": sec,
                "parent_dist": None,
                "stranger_dist": None,
                "face_movement": None,
                "body_movement": None
            })
            continue

        PROGRESS_STORE[process_id].update({
            "current": sec,
            "percent": 40 + int((sec + 1) / duration * 60)
        })

        try:
            child_roi = detect_child_and_crop(frame)
            face_score, curr_landmarks = facial_keypoints(child_roi, prev_landmarks)
            pose_kps = process_pose(child_roi)
            body_mv = calculate_body_movement(pose_kps, prev_pose)
            mov_ratio = body_mv / freeplay_movement if freeplay_movement else 0.0

            parent_dist = calculate_distance_between_objects(frame, "Child", "Adult")
            stranger_dist = calculate_distance_between_objects(frame, "Child", "Stranger")

            results.append({
                "second": sec,
                "distance_adult": parent_dist,
                "distance_stranger": stranger_dist,
                "facial_movement": face_score,
                "body_movement": mov_ratio
            })

            prev_landmarks = curr_landmarks
            prev_pose = pose_kps

        except Exception as e:
            logger.error(f"Experiment error at {sec}s: {e}", exc_info=True)
            # still append a row so CSV timestamps remain aligned
            results.append({
                "second": sec,
                "distance_adult": None,
                "distance_stranger": None,
                "facial_movement": None,
                "body_movement": None
            })

    cap.release()
    return pd.DataFrame(results)

def apply_classes(df, timestamp_start, timestamp_end,
                  distance_model_path='distance_classifier.pkl',
                  fear_model_path='fear_classifier.pkl',
                  freeze_model_path='freeze_classifier.pkl'):

    # Load models
    distance_clf = joblib.load(distance_model_path)
    fear_clf     = joblib.load(fear_model_path)
    freeze_clf   = joblib.load(freeze_model_path)

    # 1) Initialize outputs
    df['proximity to parent']   = None
    df['proximity to stranger'] = None
    df['fear']                  = None
    df['freeze']                = pd.Series([pd.NA] * len(df), dtype="Int64")

    # 2) Distance → proximity classes
    valid_mask = df[['distance_adult','body_movement','facial_movement']].notnull().all(axis=1)
    preds_parent = distance_clf.predict(df.loc[valid_mask, ['distance_adult']])
    df.loc[valid_mask, 'proximity to parent'] = preds_parent
    df.loc[valid_mask, 'proximity to stranger'] = pd.Series(preds_parent).map({0:2, 1:1, 2:0}).values

    # 3) Fear classifier
    fear_cols = ['proximity to parent','proximity to stranger','body_movement','facial_movement']
    fear_mask = df[fear_cols].notnull().all(axis=1)
    df.loc[fear_mask, 'fear'] = fear_clf.predict(df.loc[fear_mask, fear_cols])

    # 4) Build pairwise DataFrame (includes 'second')
    df1 = df.iloc[:-1].reset_index(drop=True).add_suffix('_1')
    df2 = df.iloc[1:].reset_index(drop=True).add_suffix('_2')
    df_pairs = pd.concat([df1, df2], axis=1)

    # 5) Filter pairs where both fears > 0
    mask = (df_pairs['fear_1'] > 0) & (df_pairs['fear_2'] > 0)
    df_filtered = df_pairs[mask].copy()
    df_filtered['body_movement_avg'] = (df_filtered['body_movement_1'] + df_filtered['body_movement_2']) / 2

    # 6) Predict freeze and backfill to both seconds
    if not df_filtered.empty:
        df_filtered['freeze'] = freeze_clf.predict(df_filtered[['body_movement_avg']])
        for _, row in df_filtered.iterrows():
            for sec_col in ('second_1', 'second_2'):
                sec = int(row[sec_col])
                idx = df.index[df['second'] == sec][0]
                current = df.at[idx, 'freeze']
                if not (pd.notna(current) and current == 1):
                    df.at[idx, 'freeze'] = row['freeze']

    # 7) Add timestamps column based on timestamp_start and 'second'
    time_format = '%H:%M:%S'
    ts_start = datetime.datetime.strptime(timestamp_start, time_format)
    df['timestamp'] = df['second'].apply(
        lambda x: (ts_start + datetime.timedelta(seconds=int(x))).time().strftime(time_format)
    )

    # 8) Return only the final columns
    return df[['timestamp', 'second', 'proximity to parent', 'proximity to stranger', 'fear', 'freeze']]

async def process_video_async(process_id: str, video_path: Path, session_dir: Path,
                              timestamp1: str, timestamp2: str, timestamp3: str, temp_dir: Path):

    if PROGRESS_STORE.get(process_id, {}).get("started"):
        return
    
    # Initialize progress tracking
    PROGRESS_STORE[process_id] = {
        "started": True,
        "status":  "processing",
        "percent": 0,
        "message": "Initializing",
        "result":  None,
        "error":   None
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
    
    
    try:
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
        freeplay_movement = await asyncio.to_thread(
            process_freeplay,
            process_id,
            freeplay_video
        )

        # Process experiment segment in a thread
        PROGRESS_STORE[process_id].update({
            "message": "Analyzing experiment",
            "percent": 40
        })
        result_df = await asyncio.to_thread(
            process_experiment,
            process_id,
            experiment_video,
            freeplay_movement
        )

        final_df = apply_classes(result_df, timestamp2, timestamp3)
        
        result_path = session_dir / "analysis.csv"
        final_df.to_csv(result_path, index=False)
        os.sync()

        PROGRESS_STORE[process_id].update({
            "status": "completed",
            "result": str(result_path),
            "percent": 100,
            "message": "Analysis complete"
        })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        PROGRESS_STORE[process_id].update({
            "status": "error",
            "error": str(e),
            "percent": 100
        })
    
    finally:
        if video_path.exists():
            video_path.unlink()
            
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
    # 1) Generate IDs & dirs
    process_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.mkdtemp())
    session_dir = OUTPUT_DIR / f"session_{process_id}"
    session_dir.mkdir(exist_ok=True)

    # 2) Seed progress (so /api/progress can pick it up immediately)
    PROGRESS_STORE[process_id] = {
        "started": False,
        "status":  "queued",
        "percent": 0,
        "message": "Queued for processing",
        "result":  None,
        "error":   None
    }

    # 3) Save the upload
    video_path = temp_dir / video.filename
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # 4) Kick off the async worker on the loop directly
    asyncio.create_task(
        process_video_async(
            process_id, video_path, session_dir,
            timestamp1, timestamp2, timestamp3, temp_dir
        )
    )

    # 5) Return the process_id immediately
    return {"process_id": process_id}
    
@app.get("/api/progress/{process_id}")
async def progress_stream(process_id: str):
    async def event_generator():
        last = {}
        while True:
            if process_id in PROGRESS_STORE:
                current = PROGRESS_STORE[process_id]
                if current != last:
                    last = current.copy()    # snapshot instead of alias
                    yield f"data: {json.dumps(current)}\n\n"
                if current["status"] in ["completed", "error", "cancelled"]:
                    break
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive"   # ensure the stream stays open
        }
    )

@app.get("/api/results/{process_id}")
async def results(process_id: str):
    if process_id not in PROGRESS_STORE:
        raise HTTPException(404, detail="Process ID not found")
    
    status = PROGRESS_STORE[process_id]
    
    if status["status"] == "completed":
        csv_path = Path(status["result"])
        try:
            # Validate file exists and is readable
            if not csv_path.exists() or csv_path.stat().st_size == 0:
                raise FileNotFoundError("Result file missing or empty")
                
            return FileResponse(
                csv_path,
                media_type="text/csv",
                filename="stranger_danger_analysis.csv",
                headers={"X-Analysis-Complete": "true"}
            )
        except Exception as e:
            logger.error(f"Results delivery failed: {str(e)}")
            raise HTTPException(500, detail="Results generation failed")
    
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


