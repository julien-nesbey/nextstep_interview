"""
Improved Interview Application with AI-powered question generation and emotion analysis.
"""

import eventlet

# Apply monkey patching first
eventlet.monkey_patch(os=True, select=True, socket=True, thread=True, time=True)

import base64  # noqa
import cv2  # noqa
import io  # noqa
import logging  # noqa
import numpy as np  # noqa
import os  # noqa
import sys  # noqa
import time  # noqa
from concurrent.futures import ThreadPoolExecutor  # noqa
from flask import Flask, request  # noqa
from flask_socketio import SocketIO  # noqa
from functools import lru_cache  # noqa
from logging.handlers import RotatingFileHandler  # noqa
from PIL import Image  # noqa
from queue import Queue  # noqa
from threading import Thread, Lock  # noqa
from dotenv import load_dotenv  # noqa
from datetime import datetime  # noqa
import json  # noqa

# Local imports
from prompts import question_prompt, analysis_prompt  # noqa
from utils.functions import (  # noqa
    startInterview,  # noqa
    saveInterviewData,  # noqa
    endInterview,  # noqa
    getInterviewData,  # noqa
    saveInterviewAnalysis,  # noqa
    detect_rephrase,  # noqa
)  # noqa

# Analysis imports

# Load environment variables
load_dotenv()

# Configure logging
log_handler = RotatingFileHandler("app.log", maxBytes=10000000, backupCount=5)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger("interview_app")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logger.addHandler(console_handler)

# Flask and SocketIO setup
app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=5e7,
    logger=True,
    engineio_logger=True,
)

# Constants
MAX_INTERVIEW_ROUNDS = 2
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
FACE_PROTOTXT = "models/face_detection/deploy.prototxt"
FACE_MODEL = "models/face_detection/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Emotion mapping
EMOTION_MAP = {
    "happy": "positive",
    "sad": "negative",
    "angry": "negative",
    "fear": "nervous",
    "disgust": "negative",
    "surprise": "excited",
    "neutral": "neutral",
}

# Initialize LLM client
client = None
client_pool = []
MAX_CLIENTS = 3


def init_llm_client():
    """Initialize LLM client with proper error handling and retries"""
    global client, client_pool

    if not LLM_API_KEY:
        logger.warning("No LLM API key found, LLM features will be unavailable")
        return None

    try:
        from together import Together

        # Create main client
        main_client = Together(api_key=LLM_API_KEY)

        # Test the client with a simple completion
        test_response = main_client.chat.completions.create(
            model=LLM_MODEL or "togethercomputer/llama-2-7b-chat",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            temperature=0.7,
        )

        if test_response and test_response.choices:
            logger.info("Together AI client initialized and tested successfully")
            client = main_client

            # Initialize client pool for concurrent operations
            for i in range(MAX_CLIENTS):
                try:
                    pool_client = Together(api_key=LLM_API_KEY)
                    client_pool.append(pool_client)
                    logger.debug(f"Added client {i + 1} to client pool")
                except Exception as e:
                    logger.error(f"Failed to add client {i + 1} to pool: {str(e)}")

            return client
        else:
            logger.error("LLM client test failed")
            return None

    except Exception as e:
        logger.error(f"Failed to initialize Together client: {str(e)}")
        return None


def get_llm_client():
    """Get a client from the pool or the main client if pool is empty"""
    if client_pool:
        return client_pool[hash(time.time()) % len(client_pool)]
    return client


# Global state
class InterviewState:
    def __init__(self):
        # Core state
        self.interview_data = {}  # interview_id -> interview data
        self.question_counts = {}  # interview_id -> count
        self.connected_clients = {}  # client_id -> client info
        self.tts_cache = {}  # text -> audio bytes

        # Locks
        self.interview_data_lock = Lock()
        self.question_counts_lock = Lock()
        self.clients_lock = Lock()
        self.tts_cache_lock = Lock()

        # Models
        self.face_net = None
        self.emotion_model = None
        self.emotion_model_lock = Lock()

        # Queues
        self.frame_queue = Queue(maxsize=100)
        self.question_queue = Queue(maxsize=20)

        # Workers
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.emotion_worker_threads = []


state = InterviewState()


# Initialize models
def load_face_detector():
    """Load face detection model with proper error handling"""
    if state.face_net is not None:
        return state.face_net

    try:
        if not os.path.exists(FACE_PROTOTXT) or not os.path.exists(FACE_MODEL):
            logger.error("Face detection model files missing")
            return None

        net = cv2.dnn.readNetFromCaffe(FACE_PROTOTXT, FACE_MODEL)
        logger.info(f"Loaded face detector: {net.getLayerNames()[-1]}")

        # Use GPU if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            logger.info("Face detector using CUDA acceleration")

        # Test model
        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(test_img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        net.forward()

        state.face_net = net
        return net
    except Exception as e:
        logger.error(f"Face detector initialization failed: {str(e)}")
        return None


@lru_cache(maxsize=1)
def get_emotion_model():
    """Load emotion detection model with proper caching"""
    with state.emotion_model_lock:
        if state.emotion_model is None:
            try:
                from transformers import pipeline

                logger.info("Initializing emotion model")
                state.emotion_model = pipeline(
                    "image-classification",
                    model="Rajaram1996/FacialEmoRecog",
                    use_fast=True,
                )
                logger.info("Emotion model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize emotion model: {str(e)}")

                # Fallback model
                class DummyModel:
                    def __call__(self, image):
                        return [{"label": "neutral", "score": 1.0}]

                state.emotion_model = DummyModel()

    return state.emotion_model


# TTS functions
@lru_cache(maxsize=200)
def get_cached_audio(text):
    """Get audio from cache or generate it using LRU cache

    This uses Python's lru_cache for automatic cache management rather than manual caching.
    The maxsize parameter sets the maximum number of entries before discarding least recently used entries.
    """
    try:
        # Try Edge TTS first
        return generate_edge_tts(text)
    except Exception as e:
        logger.error(f"Edge TTS error: {str(e)}")

        try:
            # Fallback to gTTS
            return generate_gtts(text)
        except Exception as fallback_error:
            logger.error(f"Fallback TTS error: {str(fallback_error)}")
            return b""


def generate_edge_tts(text):
    """Generate audio using Edge TTS via subprocess"""
    import subprocess
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w"
        ) as text_file:
            text_path = text_file.name
            text_file.write(text)

        cmd = [
            "edge-tts",
            "--voice",
            "en-US-MichelleNeural",
            "--rate",
            "+15%",
            "--file",
            text_path,
            "--write-media",
            temp_path,
        ]

        result = subprocess.run(cmd, timeout=10, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Edge TTS failed: {result.stderr}")

        with open(temp_path, "rb") as f:
            audio_data = f.read()

        os.unlink(temp_path)
        os.unlink(text_path)

        return audio_data
    except Exception as e:
        logger.error(f"Edge TTS error: {str(e)}")
        raise


def generate_gtts(text):
    """Generate audio using gTTS as fallback"""
    from gtts import gTTS

    audio_stream = io.BytesIO()
    tts = gTTS(text=text, lang="en")
    tts.write_to_fp(audio_stream)
    return audio_stream.getvalue()


# Emotion analysis
def analyze_emotion(frame):
    """Analyze emotions in a frame with optimized face detection and processing"""
    start_time = time.time()

    try:
        if frame is None or frame.size == 0:
            return {
                "label": "invalid_frame",
                "confidence": 0.0,
                "timestamp": time.time(),
            }

        frame_copy = frame.copy()

        # Performance optimization: resize large frames
        height, width = frame_copy.shape[:2]
        max_dimension = 640

        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_height, new_width = int(height * scale), int(width * scale)
            frame_copy = cv2.resize(frame_copy, (new_width, new_height))

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

        # First try Haar cascade for speed
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # If Haar fails, try DNN
        if len(faces) == 0 and state.face_net is not None:
            try:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame_copy, (300, 300)),
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0),
                )
                state.face_net.setInput(blob)
                detections = state.face_net.forward()

                h, w = frame_copy.shape[:2]
                faces_dnn = []

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x, y, x2, y2 = box.astype(int)

                        # Ensure coordinates are valid
                        x = max(0, min(x, w - 1))
                        y = max(0, min(y, h - 1))
                        x2 = max(0, min(x2, w - 1))
                        y2 = max(0, min(y2, h - 1))

                        if x2 <= x or y2 <= y:
                            continue

                        faces_dnn.append((x, y, x2 - x, y2 - y))

                faces = np.array(faces_dnn) if faces_dnn else np.array([])
            except Exception as e:
                logger.error(f"DNN face detection failed: {str(e)}")
                # Continue with whatever faces we have (or empty array)

        # No faces detected
        if len(faces) == 0:
            return {"label": "no_face", "confidence": 0.0, "timestamp": time.time()}

        # Get largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Add padding to face region
        padding = int(w * 0.1)
        y_start = max(0, y - padding)
        y_end = min(frame_copy.shape[0], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(frame_copy.shape[1], x + w + padding)

        face_roi = frame_copy[y_start:y_end, x_start:x_end]

        if face_roi.size == 0:
            return {
                "label": "invalid_face",
                "confidence": 0.0,
                "timestamp": time.time(),
            }

        # Prepare image for emotion model
        resized = cv2.resize(face_roi, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Run emotion analysis
        try:
            model = get_emotion_model()
            results = model(pil_img)

            # Process results
            processed = sorted(
                [
                    {
                        "label": EMOTION_MAP.get(r["label"].lower(), "neutral"),
                        "confidence": float(r["score"]),
                        "original_label": r["label"].lower(),
                    }
                    for r in results
                ],
                key=lambda x: x["confidence"],
                reverse=True,
            )

            elapsed = time.time() - start_time

            return {
                "label": processed[0]["label"],
                "confidence": processed[0]["confidence"],
                "original_label": processed[0]["original_label"],
                "timestamp": time.time(),
                "processing_time": elapsed,
                "face_location": [int(x), int(y), int(w), int(h)],
            }
        except Exception as e:
            logger.error(f"Emotion model inference error: {str(e)}")
            return {"label": "error", "confidence": 0.0, "timestamp": time.time()}

    except Exception as e:
        logger.error(f"Emotion analysis error: {str(e)}")
        return {"label": "error", "confidence": 0.0, "timestamp": time.time()}


# LLM functions
def generate_question(role, user_id="", interview_id=""):
    """Generate interview questions with improved handling"""
    try:
        logger.info(f"Generating question for role: {role}, interview: {interview_id}")

        # Get current round
        current_round = 1
        with state.question_counts_lock:
            if interview_id in state.question_counts:
                current_round = state.question_counts[interview_id] + 1

        is_final_round = current_round == MAX_INTERVIEW_ROUNDS
        logger.info(
            f"Question for round {current_round}/{MAX_INTERVIEW_ROUNDS}, is_final={is_final_round}"
        )

        # Get previous answers
        answers = []
        prev_question = ""

        with state.interview_data_lock:
            if interview_id in state.interview_data:
                answers = [
                    a["answer"]
                    for a in state.interview_data[interview_id]["user_answers"]
                ]
                if state.interview_data[interview_id]["ai_questions"]:
                    prev_question = state.interview_data[interview_id]["ai_questions"][
                        -1
                    ]["text"]

        # Create prompt
        messages = [
            {"role": "system", "content": question_prompt},
            {"role": "user", "content": f"Role: {role}"},
            {
                "role": "user",
                "content": f"Current round: {current_round} of {MAX_INTERVIEW_ROUNDS}",
            },
        ]

        if prev_question:
            messages.append(
                {"role": "user", "content": f"Previous question: {prev_question}"}
            )

        if answers:
            context = ", ".join(answers[-3:])
            messages.append({"role": "user", "content": f"Previous answers: {context}"})

        # Generate question with LLM
        max_retries = 3
        retries = 0
        timeout_seconds = 10

        if not client or not LLM_API_KEY:
            logger.warning("API client not available, using fallback questions")
            return fallback_question(role, len(answers), current_round, is_final_round)

        while retries < max_retries:
            try:
                backoff = min(5, 2**retries)
                model_to_use = LLM_MODEL or "togethercomputer/llama-2-7b-chat"

                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=0.7,
                    timeout=timeout_seconds,
                    user=user_id or "anonymous",
                    stream=False,
                )

                if not response or not response.choices:
                    retries += 1
                    time.sleep(backoff)
                    continue

                question = response.choices[0].message.content
                question = question.replace("```json", "").replace("```", "").strip()

                if question:
                    # Store question
                    with state.interview_data_lock:
                        if interview_id not in state.interview_data:
                            state.interview_data[interview_id] = {
                                "user_answers": [],
                                "ai_questions": [],
                                "emotions": [],
                                "start_time": time.time(),
                                "user_id": user_id,
                                "role": role,
                            }

                        question_obj = {
                            "text": question,
                            "timestamp": time.time(),
                            "is_rephrased": False,
                        }

                        state.interview_data[interview_id]["ai_questions"].append(
                            question_obj
                        )

                    return {"question": question, "is_final": is_final_round}

                break

            except Exception as e:
                retries += 1
                logger.warning(f"LLM retry {retries}/{max_retries}: {str(e)}")

                if retries < max_retries:
                    time.sleep(backoff)
                else:
                    logger.error(f"LLM error after {max_retries} attempts: {str(e)}")

        # Fallback if all retries fail
        question_data = fallback_question(
            role, len(answers), current_round, is_final_round
        )

        # Store fallback question
        with state.interview_data_lock:
            if interview_id not in state.interview_data:
                state.interview_data[interview_id] = {
                    "user_answers": [],
                    "ai_questions": [],
                    "emotions": [],
                    "start_time": time.time(),
                    "user_id": user_id,
                    "role": role,
                }

            question_obj = {
                "text": question_data["question"],
                "timestamp": time.time(),
                "is_fallback": True,
            }

            state.interview_data[interview_id]["ai_questions"].append(question_obj)

        return question_data

    except Exception as e:
        logger.error(f"Question generation error: {str(e)}")
        return fallback_question(role, 0, current_round, is_final_round)


def fallback_question(role, answer_count, current_round=1, is_final=False):
    """Provide fallback questions if LLM fails"""
    fallback_questions = {
        "default": [
            f"Tell me more about your experience as a {role}. What skills do you bring to this position?",
            "What's your greatest professional achievement so far?",
            "How do you handle tight deadlines and pressure in your work?",
            "Tell me about a time you had to learn a new skill quickly for a project.",
            "How do you prefer to receive feedback on your work?",
        ],
        "final": [
            f"This concludes our interview. Is there anything else you'd like to share about why you'd be a great fit for this {role} position?",
            "As we wrap up this interview, do you have any questions for me about the position or what it's like to work here?",
            "Thank you for your time today. Final question: what makes you the ideal candidate for this role compared to others with similar qualifications?",
            "We're at the end of our interview now. Is there anything important about your experience or skills that we haven't covered yet?",
            "To conclude our interview today, how soon would you be able to start if selected for this position?",
        ],
    }

    questions = (
        fallback_questions["final"]
        if is_final
        else fallback_questions.get(role.lower(), fallback_questions["default"])
    )
    index = answer_count % len(questions)

    return {"question": questions[index], "is_final": is_final}


# Worker functions
def emotion_worker():
    """Worker thread to process frames for emotion analysis"""
    while True:
        item = None
        try:
            try:
                item = state.frame_queue.get(timeout=1.0)
                if item is None:  # Exit signal
                    state.frame_queue.task_done()
                    break
            except Exception:
                continue

            client_id = item.get("client_id")
            frame_data = item.get("frame_data")
            timestamp = item.get("timestamp", time.time())
            interview_id = item.get("interview_id", "")

            with state.clients_lock:
                if client_id not in state.connected_clients:
                    state.frame_queue.task_done()
                    continue

            # Decode frame
            try:
                if isinstance(frame_data, str) and "," in frame_data:
                    img_data = base64.b64decode(frame_data.split(",")[1])
                elif isinstance(frame_data, str):
                    img_data = base64.b64decode(frame_data)
                else:
                    state.frame_queue.task_done()
                    continue

                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None or img.size == 0:
                    state.frame_queue.task_done()
                    continue

                # Analyze emotion
                result = analyze_emotion(img)
                result["timestamp"] = timestamp

                # Store result if valid
                if result["label"] not in [
                    "no_face",
                    "invalid_face",
                    "error",
                    "invalid_frame",
                ]:
                    with state.interview_data_lock:
                        if interview_id in state.interview_data:
                            state.interview_data[interview_id]["emotions"].append(
                                result
                            )

                # Send result to client
                with state.clients_lock:
                    if client_id in state.connected_clients:
                        socketio.emit("result", result, room=client_id)

            except Exception as e:
                logger.error(f"Frame processing error: {str(e)}")

        except Exception as e:
            logger.error(f"Emotion worker error: {str(e)}")
        finally:
            if item is not None:
                state.frame_queue.task_done()


def question_worker():
    """Worker thread to process question generation"""
    while True:
        item = None
        try:
            try:
                item = state.question_queue.get(timeout=1.0)
                if item is None:  # Exit signal
                    state.question_queue.task_done()
                    break
            except Exception:
                continue

            client_id = item.get("client_id")
            role = item.get("role")
            user_id = item.get("user_id", "")
            interview_id = item.get("interview_id", "")
            is_rephrased = item.get("is_rephrased", False)
            is_final_round = item.get("is_final_round", False)

            with state.clients_lock:
                if client_id not in state.connected_clients:
                    state.question_queue.task_done()
                    continue

            # Check if interview already completed
            with state.question_counts_lock:
                if not is_rephrased and interview_id in state.question_counts:
                    current_count = state.question_counts[interview_id]
                    if current_count >= MAX_INTERVIEW_ROUNDS:
                        # End the interview instead of just continuing
                        logger.info(
                            f"Interview {interview_id} complete, ending interview from question worker"
                        )
                        state.executor.submit(
                            enhanced_end_interview,
                            interview_id,
                            client_id,
                            is_final_round=True,
                        )
                        state.question_queue.task_done()
                        continue

            # For rephrasing, use the current question but reworded
            if is_rephrased:
                current_question = item.get("current_question", "")
                if not current_question:
                    # Find the most recent question
                    with state.interview_data_lock:
                        if (
                            interview_id in state.interview_data
                            and state.interview_data[interview_id]["ai_questions"]
                        ):
                            current_question = state.interview_data[interview_id][
                                "ai_questions"
                            ][-1]["text"]

                if current_question:
                    # Generate rephrased version
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an interview assistant. Rephrase the following question in a clearer way without changing its meaning.",
                        },
                        {
                            "role": "user",
                            "content": f"Please rephrase this question: {current_question}",
                        },
                    ]

                    try:
                        if client and LLM_API_KEY:
                            model_to_use = (
                                LLM_MODEL or "togethercomputer/llama-2-7b-chat"
                            )
                            response = client.chat.completions.create(
                                model=model_to_use,
                                messages=messages,
                                temperature=0.7,
                                timeout=10,
                                user=user_id or "anonymous",
                                stream=False,
                            )

                            if response and response.choices:
                                rephrased_question = response.choices[
                                    0
                                ].message.content.strip()
                            else:
                                rephrased_question = (
                                    f"Let me rephrase that. {current_question}"
                                )
                        else:
                            rephrased_question = (
                                f"Let me rephrase that. {current_question}"
                            )
                    except Exception as e:
                        logger.error(f"Error rephrasing question: {str(e)}")
                        rephrased_question = f"Let me clarify. {current_question}"

                    # Get current round
                    with state.question_counts_lock:
                        current_round = state.question_counts.get(interview_id, 1)

                    # Store rephrased question
                    with state.interview_data_lock:
                        if interview_id in state.interview_data:
                            question_obj = {
                                "text": rephrased_question,
                                "timestamp": time.time(),
                                "is_rephrased": True,
                            }
                            state.interview_data[interview_id]["ai_questions"].append(
                                question_obj
                            )

                    # Generate audio
                    audio_bytes = get_cached_audio(rephrased_question)

                    # Send to client
                    response_data = {
                        "question": rephrased_question,
                        "audio": base64.b64encode(audio_bytes).decode("utf-8")
                        if audio_bytes
                        else None,
                        "timestamp": time.time(),
                        "currentRound": current_round,
                        "totalRounds": MAX_INTERVIEW_ROUNDS,
                        "isRephrased": True,
                        "isFinal": current_round == MAX_INTERVIEW_ROUNDS,
                    }

                    socketio.emit("question", response_data, room=client_id)
                    state.question_queue.task_done()
                    continue

            # Normal question generation
            question_data = generate_question(role, user_id, interview_id)

            if "error" in question_data:
                socketio.emit(
                    "error", {"message": question_data["error"]}, room=client_id
                )
                state.question_queue.task_done()
                continue

            question = question_data.get("question")
            is_final = question_data.get("is_final", False) or is_final_round

            if question:
                # Generate audio
                audio_bytes = get_cached_audio(question)

                # Get current round
                current_round = 1
                with state.question_counts_lock:
                    if interview_id in state.question_counts:
                        current_round = state.question_counts[interview_id] + 1
                        if current_round == MAX_INTERVIEW_ROUNDS:
                            is_final = True

                # Send question to client
                response_data = {
                    "question": question,
                    "audio": base64.b64encode(audio_bytes).decode("utf-8")
                    if audio_bytes
                    else None,
                    "timestamp": time.time(),
                    "currentRound": current_round,
                    "totalRounds": MAX_INTERVIEW_ROUNDS,
                    "isFinal": is_final,
                }

                logger.info(
                    f"Sending question for round {current_round}/{MAX_INTERVIEW_ROUNDS}, isFinal={is_final}"
                )
                socketio.emit("question", response_data, room=client_id)
            else:
                socketio.emit(
                    "error", {"message": "Failed to generate question."}, room=client_id
                )

        except Exception as e:
            logger.error(f"Question worker error: {str(e)}")
        finally:
            if item is not None:
                state.question_queue.task_done()


# Maintenance functions
def watchdog_thread():
    """Monitor application health and clean up inactive clients"""
    while True:
        try:
            current_time = datetime.now()

            with state.clients_lock:
                disconnected = []
                for client_id, data in state.connected_clients.items():
                    if isinstance(data, dict) and "last_active" in data:
                        if (
                            current_time - data["last_active"]
                        ).total_seconds() > 300:  # 5 minutes
                            disconnected.append(client_id)

                for client_id in disconnected:
                    logger.info(f"Watchdog removing inactive client: {client_id}")
                    state.connected_clients.pop(client_id)

            # Check queue sizes
            if state.question_queue.qsize() > 10:
                logger.warning(
                    f"Question queue is backing up: {state.question_queue.qsize()} items"
                )

            if state.frame_queue.qsize() > 50:
                logger.warning(
                    f"Frame queue is backing up: {state.frame_queue.qsize()} items"
                )

            # Clean memory if needed
            if len(state.tts_cache) > 100:
                with state.tts_cache_lock:
                    logger.info(f"Clearing TTS cache ({len(state.tts_cache)} items)")
                    state.tts_cache.clear()

        except Exception as e:
            logger.error(f"Watchdog error: {str(e)}")

        time.sleep(60)


def cleanup():
    """Clean up resources on shutdown"""
    logger.info("Application shutting down, cleaning up resources...")

    # Stop workers
    logger.info("Stopping worker threads...")
    for _ in range(len(state.emotion_worker_threads)):
        try:
            state.frame_queue.put(None, block=False)
        except Exception as e:
            logger.error(f"Error sending stop signal to emotion worker: {str(e)}")

    for _ in range(2):  # Number of question workers
        try:
            state.question_queue.put(None, block=False)
        except Exception as e:
            logger.error(f"Error sending stop signal to question worker: {str(e)}")

    # Wait for queues to empty (with timeout)
    try:
        logger.info("Waiting for queues to empty...")
        frame_queue_empty = state.frame_queue.empty()
        question_queue_empty = state.question_queue.empty()

        wait_start = time.time()
        while (
            not frame_queue_empty or not question_queue_empty
        ) and time.time() - wait_start < 5:
            time.sleep(0.5)
            frame_queue_empty = state.frame_queue.empty()
            question_queue_empty = state.question_queue.empty()

        logger.info(
            f"Queue drain complete. Frame queue empty: {frame_queue_empty}, Question queue empty: {question_queue_empty}"
        )
    except Exception as e:
        logger.error(f"Error waiting for queues to empty: {str(e)}")

    # Release model resources
    logger.info("Releasing model resources...")
    try:
        state.face_net = None
        state.emotion_model = None
    except Exception as e:
        logger.error(f"Error releasing models: {str(e)}")

    # Clear caches (no longer needed with new caching mechanism)

    # Shutdown executor with a timeout
    logger.info("Shutting down thread pool...")
    try:
        state.executor.shutdown(wait=True, cancel_futures=True)
    except TypeError:
        # For older Python versions without cancel_futures
        state.executor.shutdown(wait=False)
    except Exception as e:
        logger.error(f"Error shutting down executor: {str(e)}")

    # Perform any remaining cleanup
    try:
        import gc

        gc.collect()
    except Exception:
        pass

    logger.info("Cleanup complete")


# Initialize and start application
def start_workers():
    """Start all worker threads for processing frames and questions"""
    for i in range(2):
        worker = eventlet.spawn(question_worker)
        logger.info(f"Started question worker {i}")

    for i in range(3):
        worker = eventlet.spawn(emotion_worker)
        logger.info(f"Started emotion worker {i}")
        state.emotion_worker_threads.append(worker)

    # Start watchdog
    watchdog = Thread(target=watchdog_thread, daemon=True, name="Watchdog")
    watchdog.start()
    logger.info("Started watchdog thread")


# Socket event handlers
@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    client_id = request.sid

    with state.clients_lock:
        state.connected_clients[client_id] = {
            "connected_at": datetime.now(),
            "last_active": datetime.now(),
            "frame_count": 0,
            "frame_timestamp": time.time(),
        }

    logger.info(f"Client connected: {client_id}")

    socketio.emit(
        "connection_status",
        {"status": "connected", "clientId": client_id},
        room=client_id,
    )


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid

    with state.clients_lock:
        if client_id in state.connected_clients:
            state.connected_clients.pop(client_id)

    logger.info(f"Client disconnected: {client_id}")


@socketio.on("frame")
def handle_frame(data):
    """Handle incoming video frames"""
    client_id = request.sid

    try:
        with state.clients_lock:
            if client_id not in state.connected_clients:
                return

            client_data = state.connected_clients[client_id]
            current_time = time.time()

            if not isinstance(client_data, dict):
                client_data = state.connected_clients[client_id] = {}

            client_data.setdefault("last_active", datetime.now())
            client_data.setdefault("frame_count", 0)
            client_data.setdefault("frame_timestamp", 0.0)

            client_data["last_active"] = datetime.now()
            client_data["frame_count"] += 1

            # Rate limit frames
            if current_time - client_data["frame_timestamp"] < 0.2:
                return

            client_data["frame_timestamp"] = current_time

            interview_id = client_data.get("interview_id", "")

        # Queue frame for processing
        state.frame_queue.put(
            {
                "client_id": client_id,
                "frame_data": data,
                "timestamp": current_time,
                "interview_id": interview_id,
            }
        )

    except Exception as e:
        logger.error(f"Frame handler error: {str(e)}")


@socketio.on("start_interview")
def handle_start_interview(data):
    """Handle interview start request"""
    client_id = request.sid

    try:
        if not isinstance(data, dict):
            raise ValueError("Invalid data format - expected dictionary")

        role = data.get("role", "")
        user_id = data.get("userId", "")

        if not role or not user_id:
            raise ValueError("Missing required parameters: role or userId")

        logger.info(f"Starting interview for {user_id} ({role}) - Client: {client_id}")

        # Start interview in database
        interview_id = startInterview(user_id, role)

        if not interview_id:
            raise RuntimeError("Failed to get valid interview ID")

        # Initialize interview state
        with state.clients_lock:
            state.connected_clients[client_id] = {
                "user_id": user_id,
                "role": role,
                "interview_id": interview_id,
                "active": True,
                "last_active": datetime.now(),
                "frame_count": 0,
                "frame_timestamp": time.time(),
            }

        # Initialize interview data
        with state.interview_data_lock:
            state.interview_data[interview_id] = {
                "user_answers": [],
                "ai_questions": [],
                "emotions": [],
                "start_time": time.time(),
                "user_id": user_id,
                "role": role,
            }

        # Initialize question count
        with state.question_counts_lock:
            state.question_counts[interview_id] = 0

        # Queue first question
        state.question_queue.put(
            {
                "client_id": client_id,
                "role": role,
                "user_id": user_id,
                "interview_id": interview_id,
            }
        )

        # Notify client
        socketio.emit(
            "interview_started",
            {
                "userId": user_id,
                "role": role,
                "interviewId": interview_id,
                "timestamp": datetime.now().isoformat(),
            },
            room=client_id,
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        socketio.emit(
            "error", {"code": "INVALID_INPUT", "message": str(e)}, room=client_id
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        socketio.emit(
            "error",
            {"code": "SERVER_ERROR", "message": "Failed to start interview"},
            room=client_id,
        )


@socketio.on("user_answer")
def handle_user_answer(data):
    """Handle user answers with improved rephrasing detection"""
    client_id = request.sid

    if not isinstance(data, dict):
        logger.error(f"Invalid data type for user_answer: {type(data)}")
        socketio.emit("error", {"message": "Invalid data format"}, room=client_id)
        return

    with state.clients_lock:
        client_data = state.connected_clients.get(client_id, {})
        interview_id = client_data.get("interview_id", "")
        role = client_data.get("role", "")
        user_id = client_data.get("user_id", "")

    if not interview_id:
        logger.error(f"No active interview for client {client_id}")
        socketio.emit("error", {"message": "No active interview"}, room=client_id)
        return

    answer = data.get("answer", "").strip()
    speech_analysis = data.get("speechAnalysis", {})
    logger.info(f"Received answer for interview {interview_id}: {answer[:50]}...")

    if answer:
        # Check for rephrasing request
        is_rephrase_request = False
        try:
            is_rephrase_request = detect_rephrase(answer)
            logger.info(f"Rephrasing detection result: {is_rephrase_request}")
        except Exception as e:
            logger.error(f"Error detecting rephrasing request: {str(e)}")

        timestamp = time.time()
        answer_obj = {
            "answer": answer,
            "timestamp": timestamp,
            "is_rephrase_request": is_rephrase_request,
            "speech_analysis": speech_analysis,
        }

        # Initialize interview data if not exists
        with state.interview_data_lock:
            if interview_id not in state.interview_data:
                state.interview_data[interview_id] = {
                    "user_answers": [],
                    "ai_questions": [],
                    "emotions": [],
                    "start_time": timestamp,
                    "user_id": user_id,
                    "role": role,
                }

            # Store the answer (even for rephrasing requests, but mark them)
            state.interview_data[interview_id]["user_answers"].append(answer_obj)

            # Log counts
            answer_count = len(state.interview_data[interview_id]["user_answers"])
            logger.info(f"Stored answer #{answer_count} for interview {interview_id}")

            # Save to database periodically if it's not a rephrasing request
            if not is_rephrase_request and answer_count % 2 == 0:
                try:
                    questions_copy = []
                    for q in state.interview_data[interview_id]["ai_questions"]:
                        if isinstance(q, dict):
                            questions_copy.append(q["text"])
                        else:
                            questions_copy.append(q)

                    answers_copy = []
                    for a in state.interview_data[interview_id]["user_answers"]:
                        if isinstance(a, dict) and not a.get(
                            "is_rephrase_request", False
                        ):
                            answers_copy.append(a["answer"])
                        elif not isinstance(a, dict):
                            answers_copy.append(a)

                    # Add extra validation
                    if not questions_copy or not answers_copy:
                        logger.error("Cannot save empty questions or answers")
                    elif len(questions_copy) != len(answers_copy):
                        logger.error(
                            f"Questions and answers count mismatch: {len(questions_copy)} vs {len(answers_copy)}"
                        )
                        # Try to make them match by truncating the longer one
                        min_len = min(len(questions_copy), len(answers_copy))
                        questions_copy = questions_copy[:min_len]
                        answers_copy = answers_copy[:min_len]
                        logger.info(f"Adjusted to equal lengths: {min_len}")

                    # Verify we have valid strings
                    for i, (q, a) in enumerate(zip(questions_copy, answers_copy)):
                        if not isinstance(q, str) or not isinstance(a, str):
                            logger.error(
                                f"Non-string data at index {i}: q_type={type(q)}, a_type={type(a)}"
                            )
                            questions_copy[i] = str(q) if not isinstance(q, str) else q
                            answers_copy[i] = str(a) if not isinstance(a, str) else a

                    # Log samples for debugging
                    if questions_copy and answers_copy:
                        logger.info(
                            f"Saving {len(questions_copy)} question-answer pairs"
                        )
                        logger.info(f"First question: {questions_copy[0][:50]}...")
                        logger.info(f"First answer: {answers_copy[0][:50]}...")

                    # Now save the data
                    save_result = saveInterviewData(
                        interview_id, questions_copy, answers_copy
                    )
                    if save_result:
                        logger.info(
                            f"Successfully saved interview data for {interview_id}"
                        )
                    else:
                        logger.error("Failed to save interview data (returned False)")
                except Exception as e:
                    logger.error(f"Error saving interview data: {str(e)}")
                    # Try a backup approach - direct API call
                    try:
                        from gql import Client, gql
                        from gql.transport.requests import RequestsHTTPTransport

                        transport = RequestsHTTPTransport(
                            url="http://51.20.143.187/graphql",
                            headers={"Content-Type": "application/json"},
                            use_json=True,
                        )

                        backup_client = Client(transport=transport)

                        mutation = gql("""
                        mutation SaveInterviewData($data: SaveInterviewDataInput!) {
                          saveInterviewData(interviewData: $data)
                        }
                        """)

                        result = backup_client.execute(
                            mutation,
                            variable_values={
                                "data": {
                                    "interviewId": interview_id,
                                    "question": questions_copy[
                                        :5
                                    ],  # Limit to first 5 pairs for backup
                                    "answer": answers_copy[:5],
                                }
                            },
                        )
                        logger.info(f"Backup save attempt result: {result}")
                    except Exception as backup_error:
                        logger.error(
                            f"Backup save attempt also failed: {str(backup_error)}"
                        )

        # Send acknowledgment
        socketio.emit(
            "answer_received",
            {
                "answerCount": answer_count,
                "timestamp": timestamp,
                "isRephraseRequest": is_rephrase_request,
            },
            room=client_id,
        )

        # Check if interview should continue
        should_continue = True
        is_final_round = False

        with state.question_counts_lock:
            if interview_id in state.question_counts:
                # Do NOT increment counter for rephrasing requests
                if not is_rephrase_request:
                    state.question_counts[interview_id] += 1

                current_count = state.question_counts[interview_id]
                logger.info(
                    f"Interview {interview_id} question count: {current_count}/{MAX_INTERVIEW_ROUNDS}"
                )

                if current_count >= MAX_INTERVIEW_ROUNDS:
                    should_continue = False
                    is_final_round = True
                    logger.info(
                        f"Reached max rounds ({MAX_INTERVIEW_ROUNDS}), ending interview after this answer"
                    )
            else:
                # First question
                state.question_counts[interview_id] = 1

        if should_continue:
            if is_rephrase_request:
                # For rephrasing, find the current question
                with state.interview_data_lock:
                    current_question = ""
                    if (
                        interview_id in state.interview_data
                        and state.interview_data[interview_id]["ai_questions"]
                    ):
                        last_question = state.interview_data[interview_id][
                            "ai_questions"
                        ][-1]
                        if isinstance(last_question, dict):
                            current_question = last_question["text"]
                        else:
                            current_question = last_question

                # Queue rephrased question
                state.question_queue.put(
                    {
                        "client_id": client_id,
                        "role": role,
                        "user_id": user_id,
                        "interview_id": interview_id,
                        "is_rephrased": True,
                        "current_question": current_question,
                    }
                )
            else:
                # Check if next question will be the final one
                next_is_final = False
                with state.question_counts_lock:
                    if interview_id in state.question_counts:
                        next_is_final = (
                            state.question_counts[interview_id] + 1
                            >= MAX_INTERVIEW_ROUNDS
                        )

                # Normal flow - queue next question
                state.question_queue.put(
                    {
                        "client_id": client_id,
                        "role": role,
                        "user_id": user_id,
                        "interview_id": interview_id,
                        "is_final_round": next_is_final,
                    }
                )
        else:
            # Interview complete - end it
            logger.info(
                f"Interview {interview_id} complete, ending interview from user_answer"
            )

            def execute_interview_end():
                try:
                    logger.info(
                        f"Starting enhanced_end_interview for {interview_id}, is_final_round={is_final_round}"
                    )
                    result = enhanced_end_interview(
                        interview_id, client_id, is_final_round=is_final_round
                    )
                    logger.info(
                        f"enhanced_end_interview completed with result: {result}"
                    )
                except Exception as e:
                    logger.error(f"Error ending interview: {str(e)}")

            state.executor.submit(execute_interview_end)
    else:
        logger.warning(f"Received empty answer from client {client_id}")
        socketio.emit(
            "error",
            {"message": "Please provide an answer before continuing."},
            room=client_id,
        )


# Analysis functions
def generate_interview_analysis(interview_id, role, user_id):
    """Generate comprehensive analysis of the completed interview with all data"""
    try:
        logger.info(f"Generating analysis for interview: {interview_id}, role: {role}")

        questions = []
        answers = []
        emotions = []
        speech_analytics = []

        # Try to get data from memory first
        with state.interview_data_lock:
            if interview_id in state.interview_data:
                questions = state.interview_data[interview_id]["ai_questions"]
                answers = state.interview_data[interview_id]["user_answers"]
                emotions = state.interview_data[interview_id]["emotions"]

                # Extract speech analytics from answers
                speech_analytics = [
                    a.get("speech_analysis", {})
                    for a in answers
                    if isinstance(a, dict) and "speech_analysis" in a
                ]

                logger.info(
                    f"Found interview data in memory: {len(questions)} questions, {len(answers)} answers, {len(speech_analytics)} speech analytics"
                )

        # If not in memory, try database
        if not questions or not answers:
            interview_data = getInterviewData(interview_id)

            if interview_data and interview_data.get("id"):
                # Convert database format to internal format
                questions = [
                    {"text": q["text"], "timestamp": time.time()}
                    for q in interview_data.get("questions", [])
                ]
                answers = [
                    {"answer": a["answer"], "timestamp": time.time()}
                    for a in interview_data.get("responses", [])
                ]
                logger.info(
                    f"Loaded interview data from database: {len(questions)} questions, {len(answers)} answers"
                )
            else:
                # Try backup file
                try:
                    backup_file = f"interview_data_{interview_id}.json"
                    if os.path.exists(backup_file):
                        with open(backup_file, "r") as f:
                            backup_data = json.load(f)
                            questions = backup_data.get("questions", [])
                            answers = backup_data.get("answers", [])
                            emotions = backup_data.get("emotions", [])
                            logger.info(
                                f"Loaded interview data from backup: {len(questions)} questions, {len(answers)} answers"
                            )
                    else:
                        return {"error": "No interview data found for analysis"}
                except Exception as e:
                    logger.error(f"Failed to load backup data: {str(e)}")
                    return {"error": "No interview data found for analysis"}

        # Validate data
        if not questions or not answers:
            return {"error": "Questions and answers cannot be empty"}

        if len(questions) != len(answers):
            logger.warning(
                f"Mismatched Q&A counts: {len(questions)} questions vs {len(answers)} answers"
            )
            # Try to match them up as best as possible
            min_len = min(len(questions), len(answers))
            questions = questions[:min_len]
            answers = answers[:min_len]

        # Process emotions
        emotion_summary = "No emotion data available."
        emotion_data = {}
        if emotions:
            emotion_counts = {}
            emotion_timeline = []
            for emotion in emotions:
                label = emotion.get("label", "unknown")
                timestamp = emotion.get("timestamp", 0)
                confidence = emotion.get("confidence", 0)
                if label not in ["no_face", "invalid_face", "error", "invalid_frame"]:
                    emotion_counts[label] = emotion_counts.get(label, 0) + 1
                    emotion_timeline.append(
                        {
                            "emotion": label,
                            "timestamp": timestamp,
                            "confidence": confidence,
                        }
                    )

            # Sort timeline by timestamp
            emotion_timeline = sorted(emotion_timeline, key=lambda x: x["timestamp"])

            # Calculate percentages and dominant emotions
            total = sum(emotion_counts.values())
            if total > 0:
                emotion_percentages = {
                    label: (count / total * 100)
                    for label, count in emotion_counts.items()
                }
                sorted_emotions = sorted(
                    emotion_percentages.items(), key=lambda x: x[1], reverse=True
                )
                dominant_emotions = (
                    [label for label, _ in sorted_emotions[:2]]
                    if len(sorted_emotions) >= 2
                    else [sorted_emotions[0][0]]
                    if sorted_emotions
                    else []
                )

                # Create detailed emotion summary
                emotion_summary = ", ".join(
                    [
                        f"{label}: {percentage:.1f}%"
                        for label, percentage in sorted_emotions
                    ]
                )

                # Enhanced emotion data for analysis
                emotion_data = {
                    "percentages": emotion_percentages,
                    "dominant_emotions": dominant_emotions,
                    "sample_count": total,
                    "timeline_summary": summarize_emotion_timeline(emotion_timeline),
                }

        # Process speech analytics
        speech_data = {}
        if speech_analytics:
            # Aggregate speech analytics data
            total_duration = sum(a.get("duration", 0) for a in speech_analytics)
            total_words = sum(a.get("wordCount", 0) for a in speech_analytics)
            total_pauses = sum(a.get("pauseCount", 0) for a in speech_analytics)
            total_stutters = sum(a.get("stutterCount", 0) for a in speech_analytics)

            # Speech emotion breakdown
            speech_emotions = {}
            for analytics in speech_analytics:
                emotion_breakdown = analytics.get("emotionBreakdown", {})
                for emotion, duration in emotion_breakdown.items():
                    speech_emotions[emotion] = (
                        speech_emotions.get(emotion, 0) + duration
                    )

            # Calculate averages
            avg_speaking_rate = (
                sum(a.get("speakingRate", 0) for a in speech_analytics)
                / len(speech_analytics)
                if speech_analytics
                else 0
            )
            avg_pause_duration = (
                sum(a.get("avgPauseDuration", 0) for a in speech_analytics)
                / len(speech_analytics)
                if speech_analytics
                else 0
            )

            # Identify patterns in silences
            all_silences = []
            for analytics in speech_analytics:
                silences = analytics.get("silences", [])
                all_silences.extend(silences)

            # Compile speech analytics summary
            speech_data = {
                "total_duration_seconds": total_duration / 1000
                if total_duration
                else 0,
                "total_words": total_words,
                "words_per_minute": (total_words / (total_duration / 60000))
                if total_duration
                else 0,
                "total_pauses": total_pauses,
                "total_stutters": total_stutters,
                "average_pause_duration_ms": avg_pause_duration,
                "average_speaking_rate": avg_speaking_rate,
                "speech_emotions": speech_emotions,
                "silence_pattern": "significant"
                if len(all_silences) > 5
                else "moderate"
                if len(all_silences) > 2
                else "minimal",
                "fluency_score": calculate_fluency_score(
                    total_words, total_stutters, total_pauses, avg_pause_duration
                ),
            }

            logger.info(
                f"Processed speech analytics: {len(speech_analytics)} samples, {total_words} words"
            )

        # Create prompt
        messages = [
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": f"Role: {role}"},
            {
                "role": "user",
                "content": f"Candidate emotion summary: {emotion_summary}",
            },
        ]

        # Add detailed emotion data if available
        if emotion_data:
            messages.append(
                {
                    "role": "user",
                    "content": f"Detailed emotion data: {json.dumps(emotion_data, indent=2)}",
                }
            )

        # Add speech analytics data if available
        if speech_data:
            messages.append(
                {
                    "role": "user",
                    "content": f"Speech analytics data: {json.dumps(speech_data, indent=2)}",
                }
            )

        # Format Q&A for analysis - Fix to ensure we get plain text for both questions and answers
        qa_content = ""
        for i, (q, a) in enumerate(zip(questions, answers)):
            q_text = q["text"] if isinstance(q, dict) and "text" in q else q
            a_text = a["answer"] if isinstance(a, dict) and "answer" in a else a
            qa_content += f"Question {i + 1}: {q_text.strip() if isinstance(q_text, str) else str(q_text)}\n"
            qa_content += f"Answer {i + 1}: {a_text.strip() if isinstance(a_text, str) else str(a_text)}\n\n"

        messages.append({"role": "user", "content": f"Interview Q&A:\n{qa_content}"})
        logger.info(f"Prepared analysis content with {len(questions)} Q&A pairs")

        # Generate analysis
        max_retries = 3
        retries = 0
        timeout_seconds = 20

        if not client or not LLM_API_KEY:
            logger.warning("API client not available, using fallback analysis")
            analysis = f"Interview Analysis for {role} position:\n\n"
            analysis += "The candidate demonstrated reasonable knowledge across the questions asked. "
            analysis += (
                "Based on the emotional analysis, the candidate appeared mostly "
                + emotion_summary
            )
            saveInterviewAnalysis(interview_id, analysis)
            return {"analysis": analysis, "emotion_summary": emotion_summary}

        while retries < max_retries:
            try:
                backoff = min(5, 2**retries)
                model_to_use = LLM_MODEL or "togethercomputer/llama-2-7b-chat"

                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=0.7,
                    timeout=timeout_seconds,
                    user=user_id or "anonymous",
                    stream=False,
                )

                if not response or not response.choices:
                    retries += 1
                    time.sleep(backoff)
                    continue

                analysis = response.choices[0].message.content
                analysis = analysis.replace("```json", "").replace("```", "").strip()

                if analysis:
                    # Save analysis
                    saveInterviewAnalysis(interview_id, analysis)
                    return {"analysis": analysis, "emotion_summary": emotion_summary}

                break

            except Exception as e:
                retries += 1
                logger.warning(
                    f"LLM retry for analysis {retries}/{max_retries}: {str(e)}"
                )

                if retries < max_retries:
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"LLM error in analysis after {max_retries} attempts: {str(e)}"
                    )

        # Fallback analysis
        analysis = f"Interview Analysis for {role} position:\n\n"
        analysis += "The candidate demonstrated reasonable knowledge across the questions asked. "
        analysis += (
            "Based on the emotional analysis, the candidate appeared mostly "
            + emotion_summary
        )
        saveInterviewAnalysis(interview_id, analysis)
        return {"analysis": analysis, "emotion_summary": emotion_summary}

    except Exception as e:
        logger.error(f"Analysis generation error: {str(e)}")
        return {"error": f"Failed to generate analysis: {str(e)}"}


def calculate_fluency_score(words, stutters, pauses, avg_pause_duration):
    """Calculate a fluency score from speech metrics (0-100)"""
    if words == 0:
        return 0

    # Base score starts at 100
    score = 100

    # Penalize for stutters
    stutter_penalty = min(40, (stutters / max(1, words / 20)) * 10)

    # Penalize for excessive pauses
    pause_penalty = min(30, (pauses / max(1, words / 15)) * 5)

    # Penalize for long pauses
    duration_penalty = min(30, avg_pause_duration / 100)

    # Calculate final score
    final_score = max(
        0, min(100, score - stutter_penalty - pause_penalty - duration_penalty)
    )

    return round(final_score, 1)


def enhanced_end_interview(interview_id, client_id, is_final_round=False):
    """Enhanced version of endInterview that also generates analysis but doesn't send it automatically"""
    try:
        logger.info(
            f"Ending interview {interview_id} for client {client_id}, is_final_round={is_final_round}"
        )

        # Get interview data
        role = None
        user_id = None

        with state.interview_data_lock:
            if interview_id in state.interview_data:
                role = state.interview_data[interview_id]["role"]
                user_id = state.interview_data[interview_id]["user_id"]

                # Format data for database save
                questions = []
                answers = []

                for q in state.interview_data[interview_id]["ai_questions"]:
                    if isinstance(q, dict):
                        questions.append(q["text"])
                    else:
                        questions.append(q)

                for a in state.interview_data[interview_id]["user_answers"]:
                    if isinstance(a, dict) and not a.get("is_rephrase_request", False):
                        answers.append(a["answer"])
                    else:
                        answers.append(a)

                # Save data
                try:
                    # Add extra validation
                    if not questions or not answers:
                        logger.error("Cannot save empty questions or answers")
                    elif len(questions) != len(answers):
                        logger.error(
                            f"Questions and answers count mismatch: {len(questions)} vs {len(answers)}"
                        )
                        # Try to make them match by truncating the longer one
                        min_len = min(len(questions), len(answers))
                        questions = questions[:min_len]
                        answers = answers[:min_len]
                        logger.info(f"Adjusted to equal lengths: {min_len}")

                    # Verify we have valid strings
                    for i, (q, a) in enumerate(zip(questions, answers)):
                        if not isinstance(q, str) or not isinstance(a, str):
                            logger.error(
                                f"Non-string data at index {i}: q_type={type(q)}, a_type={type(a)}"
                            )
                            questions[i] = str(q) if not isinstance(q, str) else q
                            answers[i] = str(a) if not isinstance(a, str) else a

                    # Log samples for debugging
                    if questions and answers:
                        logger.info(f"Saving {len(questions)} question-answer pairs")
                        logger.info(f"First question: {questions[0][:50]}...")
                        logger.info(f"First answer: {answers[0][:50]}...")

                    # Now save the data
                    save_result = saveInterviewData(interview_id, questions, answers)
                    if save_result:
                        logger.info(
                            f"Successfully saved interview data for {interview_id}"
                        )
                    else:
                        logger.error("Failed to save interview data (returned False)")
                except Exception as e:
                    logger.error(f"Error saving interview data: {str(e)}")
                    # Try a backup approach - direct API call
                    try:
                        from gql import Client, gql
                        from gql.transport.requests import RequestsHTTPTransport

                        transport = RequestsHTTPTransport(
                            url="http://51.20.143.187/graphql",
                            headers={"Content-Type": "application/json"},
                            use_json=True,
                        )

                        backup_client = Client(transport=transport)

                        mutation = gql("""
                        mutation SaveInterviewData($data: SaveInterviewDataInput!) {
                            saveInterviewData(interviewData: $data)
                        }
                        """)

                        result = backup_client.execute(
                            mutation,
                            variable_values={
                                "data": {
                                    "interviewId": interview_id,
                                    "question": questions[
                                        :5
                                    ],  # Limit to first 5 pairs for backup
                                    "answer": answers[:5],
                                }
                            },
                        )
                        logger.info(f"Backup save attempt result: {result}")
                    except Exception as backup_error:
                        logger.error(
                            f"Backup save attempt also failed: {str(backup_error)}"
                        )

        # Mark interview as ended in database
        result = endInterview(interview_id)
        logger.info(f"Database end interview result: {result}")

        # Send conclusion message
        if client_id:
            try:
                # Generate conclusion message
                conclusion_message = generate_conclusion_message(role)

                # Generate audio for conclusion
                audio_bytes = get_cached_audio(conclusion_message)

                # Send conclusion to client
                socketio.emit(
                    "interview_conclusion",
                    {
                        "message": conclusion_message,
                        "audio": base64.b64encode(audio_bytes).decode("utf-8")
                        if audio_bytes
                        else None,
                        "timestamp": time.time(),
                        "interviewId": interview_id,
                        "analysisAvailable": True,
                    },
                    room=client_id,
                )

                logger.info(f"Sent conclusion message for interview {interview_id}")

            except Exception as e:
                logger.error(f"Error generating conclusion message: {str(e)}")

        # Generate and store analysis but don't send it automatically
        analysis_result = generate_interview_analysis(interview_id, role, user_id)

        if analysis_result and "analysis" in analysis_result:
            # Store analysis in memory for retrieval via request
            with state.interview_data_lock:
                if interview_id in state.interview_data:
                    state.interview_data[interview_id]["analysis_result"] = (
                        analysis_result
                    )
                    logger.info(
                        f"Analysis for interview {interview_id} stored and ready for retrieval"
                    )

        # Send interview_ended event (without analysis)
        socketio.emit(
            "interview_ended",
            {
                "interviewId": interview_id,
                "timestamp": time.time(),
                "analysisAvailable": True,
            },
            room=client_id,
        )
        logger.info(f"Sent interview_ended event for interview {interview_id}")

        return result

    except Exception as e:
        logger.error(f"Error in enhanced_end_interview: {str(e)}")

        # Still try to send an ended event
        try:
            socketio.emit(
                "interview_ended",
                {
                    "interviewId": interview_id,
                    "timestamp": time.time(),
                    "error": str(e),
                },
                room=client_id,
            )
        except Exception:
            pass

        return False


def generate_conclusion_message(role):
    """Generate a personalized conclusion message for the interview

    Args:
        role: The role the candidate was interviewed for

    Returns:
        A conclusion message thanking the candidate
    """
    try:
        # Try to generate with LLM if available
        if client:
            try:
                conclusion_prompt = (
                    "Generate a friendly conclusion message for a candidate who just completed "
                    f"a technical interview for a {role} position. Thank them for their time, "
                    "mention that the analysis will follow shortly, and wish them well. "
                    "Keep it under 3 sentences and make it conversational."
                )

                messages = [{"role": "user", "content": conclusion_prompt}]

                response = get_llm_client().chat.completions.create(
                    model=LLM_MODEL or "togethercomputer/llama-2-7b-chat",
                    messages=messages,
                    temperature=0.7,
                    timeout=5,
                )

                if response and response.choices:
                    content = response.choices[0].message.content
                    content = content.replace("```json", "").replace("```", "").strip()
                    return content
            except Exception as e:
                logger.error(f"Error generating LLM conclusion: {str(e)}")

        # Fallback messages
        conclusions = [
            f"Thank you for completing this {role} interview! I appreciate your thoughtful responses. Your analysis will be ready in just a moment.",
            f"That concludes our {role} interview. Thanks for sharing your knowledge and experience. I'll have your results ready shortly.",
            f"Great job on completing the {role} interview! Your insights were valuable. I'm preparing your analysis now.",
            f"Thanks for participating in this {role} technical interview. Your responses have been recorded, and your analysis is being generated.",
        ]

        # Select a random conclusion
        import random

        return random.choice(conclusions)

    except Exception as e:
        logger.error(f"Error in conclusion generation: {str(e)}")
        return f"Thank you for completing the {role} interview. Your analysis will be available shortly."


def summarize_emotion_timeline(timeline):
    """Summarize emotion timeline to identify patterns and transitions"""
    if not timeline or len(timeline) < 2:
        return {"pattern": "insufficient data"}

    # Initialize variables
    segments = []
    current_segment = {
        "emotion": timeline[0]["emotion"],
        "duration": 0,
        "start": timeline[0]["timestamp"],
    }
    transitions = []
    prev_emotion = timeline[0]["emotion"]

    # Process timeline
    for entry in timeline:
        if entry["emotion"] == current_segment["emotion"]:
            # Continue current segment
            current_segment["duration"] = entry["timestamp"] - current_segment["start"]
        else:
            # End segment and start new one
            segments.append(current_segment)
            transitions.append(f"{prev_emotion} → {entry['emotion']}")
            current_segment = {
                "emotion": entry["emotion"],
                "duration": 0,
                "start": entry["timestamp"],
            }
            prev_emotion = entry["emotion"]

    # Add last segment
    segments.append(current_segment)

    # Find dominant segments
    dominant_segments = sorted(segments, key=lambda x: x["duration"], reverse=True)[:3]

    # Count transitions between emotions
    transition_counts = {}
    for t in transitions:
        transition_counts[t] = transition_counts.get(t, 0) + 1

    # Find dominant transitions
    dominant_transitions = sorted(
        transition_counts.items(), key=lambda x: x[1], reverse=True
    )[:3]

    # Calculate emotional stability (fewer transitions = more stable)
    stability_score = max(0, min(100, 100 - (len(transitions) * 100 / len(timeline))))

    return {
        "dominant_segments": [
            f"{s['emotion']} ({s['duration']:.1f}s)" for s in dominant_segments
        ],
        "transitions": [f"{t[0]} ({t[1]} times)" for t in dominant_transitions]
        if dominant_transitions
        else [],
        "emotional_stability": stability_score,
        "total_transitions": len(transitions),
        "pattern": "stable"
        if stability_score > 70
        else "variable"
        if stability_score > 40
        else "highly variable",
    }


@socketio.on("request_analysis")
def handle_analysis_request(data):
    """Handle client request for interview analysis"""
    client_id = request.sid

    if not isinstance(data, dict):
        logger.error(f"Invalid data type for request_analysis: {type(data)}")
        socketio.emit("error", {"message": "Invalid data format"}, room=client_id)
        return

    interview_id = data.get("interviewId", "")

    if not interview_id:
        logger.error("No interview ID provided in analysis request")
        socketio.emit("error", {"message": "No interview ID provided"}, room=client_id)
        return

    logger.info(
        f"Received analysis request for interview {interview_id} from client {client_id}"
    )

    # Check if we have the analysis in memory
    analysis_result = None
    with state.interview_data_lock:
        if (
            interview_id in state.interview_data
            and "analysis_result" in state.interview_data[interview_id]
        ):
            analysis_result = state.interview_data[interview_id]["analysis_result"]

    # If not in memory, try to generate it
    if not analysis_result:
        logger.info(f"Analysis not found in memory for {interview_id}, generating now")

        # Get interview data for generating analysis
        role = None
        user_id = None

        with state.interview_data_lock:
            if interview_id in state.interview_data:
                role = state.interview_data[interview_id].get("role")
                user_id = state.interview_data[interview_id].get("user_id")

        if role and user_id:
            analysis_result = generate_interview_analysis(interview_id, role, user_id)
        else:
            logger.error(
                f"Cannot generate analysis: missing role or user_id for interview {interview_id}"
            )

    # Send analysis to client if available
    if analysis_result and "analysis" in analysis_result:
        # Create audio introduction
        intro_message = "Here is your interview analysis. Thank you for participating."
        audio_bytes = get_cached_audio(intro_message)

        # Send to client
        socketio.emit(
            "interview_analysis",
            {
                "analysis": analysis_result["analysis"],
                "audio": base64.b64encode(audio_bytes).decode("utf-8")
                if audio_bytes
                else None,
                "emotionSummary": analysis_result.get(
                    "emotion_summary", "No emotion data available."
                ),
                "timestamp": time.time(),
                "interviewId": interview_id,
            },
            room=client_id,
        )
        logger.info(f"Sent analysis to client {client_id} for interview {interview_id}")
    else:
        socketio.emit(
            "error",
            {
                "code": "ANALYSIS_UNAVAILABLE",
                "message": "Analysis is not available for this interview",
                "interviewId": interview_id,
            },
            room=client_id,
        )
        logger.error(f"Analysis unavailable for interview {interview_id}")


@app.before_request
def initialize_app():
    """Ensure all models and resources are loaded before handling requests"""
    load_face_detector()
    get_emotion_model()


def main():
    """Application entry point with improved initialization"""
    try:
        # Load environment variables if not already loaded
        load_dotenv()

        # Check for required env variables
        required_vars = ["LLM_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("Some features may not work correctly")

        # Initialize LLM client
        init_llm_client()

        # Load models in sequence to avoid memory spikes
        logger.info("Loading face detector model...")
        load_face_detector()

        logger.info("Loading emotion model...")
        get_emotion_model()

        # Start workers
        start_workers()

        # Register cleanup
        import atexit

        atexit.register(cleanup)

        # Run server
        logger.info(f"Starting server on port 5000 (Debug mode: {DEBUG_MODE})")

        socketio.run(
            app,
            host="0.0.0.0",
            port=5000,
            debug=DEBUG_MODE,
            use_reloader=False,
            log_output=True,
        )
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
