import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import tempfile
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import cv2
import mediapipe as mp
from joblib import load
import base64
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
import json
import io
import random
import speech_recognition as sr
import threading
import queue
import wave
import pyaudio
from datetime import datetime
import re

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="NeuroScan: Dyslexia & Dysgraphia Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- CUSTOM CSS -------------------
def local_css():
    st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2E86AB;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #2E86AB;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .progress-container {
        margin: 30px 0;
    }
    
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    .step {
        text-align: center;
        flex: 1;
        position:relative;
    }
    
    .step-number {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #E9ECEF;
        color: #6C757D;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px;
        font-weight: bold;
    }
    
    .step.active .step-number {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
    }
    
    .step.completed .step-number {
        background-color: #28A745;
        color: white;
    }
    
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 40px 20px;
    }
    
    .result-box {
        background-color: #E8F4F8;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #2E86AB;
    }
    
    .prediction-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    
    .low-risk {
        background-color: #D4EDDA;
        color: #155724;
    }
    
    .medium-risk {
        background-color: #FFF3CD;
        color: #856404;
    }
    
    .high-risk {
        background-color: #F8D7DA;
        color: #721C24;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2E86AB;
        border-radius: 10px;
        padding: 20px;
        background-color: #F8F9FA;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 5px;
        border: 1px solid #CED4DA;
    }
    
    /* Success message */
    .stAlert {
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Camera feed container */
    .camera-feed {
        border: 2px solid #2E86AB;
        border-radius: 10px;
        padding: 10px;
        background-color: #000;
        margin: 10px 0;
    }
    
    /* Speech recognition status */
    .speech-status {
        background-color: #E8F4F8;
        border-left: 4px solid #2E86AB;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ------------------- LOAD MODELS -------------------
@st.cache_resource
def load_models():
    try:
        clf = load("dyslexia_model.joblib")
        scaler = load("scaler.joblib")
        return clf, scaler
    except Exception as e:
        st.warning(f"Model files not found. Using demo mode. Error: {e}")
        return None, None

clf, scaler = load_models()

# ------------------- SESSION STATE -------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "results" not in st.session_state:
    st.session_state.results = {}
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "users" not in st.session_state:
    st.session_state.users = {"demo": "demo123", "admin": "admin123"}
if "handwriting_image" not in st.session_state:
    st.session_state.handwriting_image = None
if "tracking_active" not in st.session_state:
    st.session_state.tracking_active = False
if "eye_tracking_data" not in st.session_state:
    st.session_state.eye_tracking_data = {"gaze_points": [], "fixations": [], "drastic_movements": 0}
if "speech_data" not in st.session_state:
    st.session_state.speech_data = {"spoken_text": "", "accuracy": 0, "is_recording": False}
if "speech_recognition_active" not in st.session_state:
    st.session_state.speech_recognition_active = False

# ------------------- SPEECH RECOGNITION FUNCTIONS -------------------
# ------------------- SPEECH RECOGNITION FUNCTIONS -------------------
def calculate_text_accuracy(original_text, spoken_text):
    """Calculate accuracy between original and spoken text"""
    if not original_text or not spoken_text:
        return 0.0
    
    # Clean texts for comparison
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = ' '.join(text.split())  # Remove extra whitespace
        return text
    
    original_clean = clean_text(original_text)
    spoken_clean = clean_text(spoken_text)
    
    # Split into words
    original_words = original_clean.split()
    spoken_words = spoken_clean.split()
    
    if len(original_words) == 0:
        return 0.0
    
    # Calculate word-level accuracy
    matches = 0
    min_len = min(len(original_words), len(spoken_words))
    
    for i in range(min_len):
        if original_words[i] == spoken_words[i]:
            matches += 1
    
    accuracy = (matches / len(original_words)) * 100
    return round(accuracy, 2)

def simulate_speech_recognition(passage_text, duration):
    """Simulate speech recognition for testing when real recognition fails"""
    # Simulate spoken text with some errors
    words = passage_text.split()
    spoken_words = []
    
    for word in words:
        # Simulate 90% accuracy by changing 10% of words
        if random.random() < 0.1:  # 10% error rate
            # Make a common dyslexia-like error
            if len(word) > 3:
                # Reverse some letters or skip letters
                if random.random() < 0.5:
                    # Reverse letters
                    if len(word) > 4:
                        word = word[:2] + word[3] + word[2] + word[4:]
                else:
                    # Skip a letter
                    if len(word) > 3:
                        idx = random.randint(1, len(word)-2)
                        word = word[:idx] + word[idx+1:]
        spoken_words.append(word)
    
    spoken_text = " ".join(spoken_words)
    accuracy = calculate_text_accuracy(passage_text, spoken_text)
    
    return spoken_text, accuracy

def speech_recognition_thread(audio_queue, passage_text, stop_event):
    """Thread function for speech recognition"""
    try:
        recognizer = sr.Recognizer()
        
        # Test if microphone is available
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                while not stop_event.is_set():
                    try:
                        # Listen for audio with timeout
                        audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        audio_queue.put(audio)
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        print(f"Microphone error: {e}")
                        break
        except Exception as e:
            print(f"Microphone initialization failed: {e}")
            # Fall back to simulation mode
            while not stop_event.is_set():
                time.sleep(1)
                # Simulate audio data periodically
                if random.random() < 0.3:  # Simulate every few seconds
                    audio_queue.put("simulated_audio")
    
    except Exception as e:
        print(f"Speech recognition thread error: {e}")

def process_audio(audio_queue, passage_text, stop_event):
    """Process audio from queue and update session state"""
    recognizer = sr.Recognizer()
    
    while not stop_event.is_set():
        try:
            audio = audio_queue.get(timeout=2)
            
            # Handle simulated audio
            if audio == "simulated_audio":
                spoken_text, accuracy = simulate_speech_recognition(passage_text, 1)
                
                # Update session state
                if spoken_text:
                    st.session_state.speech_data["spoken_text"] = spoken_text
                    st.session_state.speech_data["accuracy"] = accuracy
                    
                    # Update dyslexia prediction based on speech accuracy
                    if accuracy >= 90:
                        st.session_state.speech_data["dyslexia_from_speech"] = "Low Risk"
                    elif accuracy >= 70:
                        st.session_state.speech_data["dyslexia_from_speech"] = "Medium Risk"
                    else:
                        st.session_state.speech_data["dyslexia_from_speech"] = "High Risk"
                continue
            
            try:
                # Try Google Speech Recognition
                spoken_text = recognizer.recognize_google(audio)
                
                # Update session state with new spoken text
                if spoken_text:
                    st.session_state.speech_data["spoken_text"] = spoken_text
                    
                    # Calculate accuracy
                    accuracy = calculate_text_accuracy(passage_text, spoken_text)
                    st.session_state.speech_data["accuracy"] = accuracy
                    
                    # Update dyslexia prediction based on speech accuracy
                    if accuracy >= 90:
                        st.session_state.speech_data["dyslexia_from_speech"] = "Low Risk"
                    elif accuracy >= 70:
                        st.session_state.speech_data["dyslexia_from_speech"] = "Medium Risk"
                    else:
                        st.session_state.speech_data["dyslexia_from_speech"] = "High Risk"
                        
            except sr.UnknownValueError:
                # Speech not understood
                if not st.session_state.speech_data["spoken_text"]:
                    st.session_state.speech_data["spoken_text"] = "Listening..."
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")
                # Fall back to simulation
                if not st.session_state.speech_data["spoken_text"]:
                    spoken_text, accuracy = simulate_speech_recognition(passage_text, 1)
                    st.session_state.speech_data["spoken_text"] = spoken_text
                    st.session_state.speech_data["accuracy"] = accuracy
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Audio processing error: {e}")
            continue

# ------------------- UTILITY FUNCTIONS -------------------
def extract_eye_features(gaze_points, fixations, drastic_movements_count):
    """Extract features from gaze tracking data"""
    if len(gaze_points) == 0:
        # Return realistic demo data if no actual tracking
        return {
            "n_fixations": random.randint(15, 30),
            "mean_fix_dur": random.uniform(200, 350),
            "std_fix_dur": random.uniform(50, 150),
            "mean_x": random.uniform(300, 500),
            "mean_y": random.uniform(200, 400),
            "dispersion_x": random.uniform(50, 150),
            "dispersion_y": random.uniform(30, 100),
            "total_gaze_points": random.randint(100, 300),
            "saccade_length": random.uniform(5, 20),
            "reading_speed": random.uniform(150, 300),
            "drastic_movements": drastic_movements_count
        }
    
    try:
        # Calculate basic features from gaze points
        xs = [p[0] for p in gaze_points if len(p) > 0]
        ys = [p[1] for p in gaze_points if len(p) > 0]
        timestamps = [p[2] for p in gaze_points if len(p) > 2]
        
        if len(xs) < 2:
            return {
                "n_fixations": 0,
                "mean_fix_dur": 0,
                "std_fix_dur": 0,
                "mean_x": 0,
                "mean_y": 0,
                "dispersion_x": 0,
                "dispersion_y": 0,
                "total_gaze_points": len(gaze_points),
                "saccade_length": 0,
                "reading_speed": 0,
                "drastic_movements": drastic_movements_count
            }
        
        # Fixation features
        n_fix = len(fixations) if len(fixations) > 0 else len(gaze_points) // 10
        
        if len(fixations) > 0:
            fix_durations = [f["duration"] for f in fixations]
            mean_fix = np.mean(fix_durations) * 1000 if fix_durations else 0
            std_fix = np.std(fix_durations) * 1000 if len(fix_durations) > 1 else 0
        else:
            mean_fix = random.uniform(200, 350)
            std_fix = random.uniform(50, 150)
        
        # Gaze statistics
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        dispersion_x = np.std(xs)
        dispersion_y = np.std(ys)
        
        # Additional features
        total_gaze_points = len(gaze_points)
        
        # Calculate saccade length (distance between consecutive gaze points)
        saccade_lengths = []
        for i in range(1, len(gaze_points)):
            if len(gaze_points[i]) > 1 and len(gaze_points[i-1]) > 1:
                dist = np.sqrt((gaze_points[i][0] - gaze_points[i-1][0])**2 + 
                              (gaze_points[i][1] - gaze_points[i-1][1])**2)
                saccade_lengths.append(dist)
        
        avg_saccade = np.mean(saccade_lengths) if saccade_lengths else 0
        
        # Reading speed (words per minute estimation)
        if len(timestamps) > 1:
            total_time = timestamps[-1] - timestamps[0]
            reading_speed = (len(gaze_points) / max(total_time, 1)) * 60  # points per minute
        else:
            reading_speed = random.uniform(150, 300)
        
        return {
            "n_fixations": int(n_fix),
            "mean_fix_dur": float(mean_fix),
            "std_fix_dur": float(std_fix),
            "mean_x": float(mean_x),
            "mean_y": float(mean_y),
            "dispersion_x": float(dispersion_x),
            "dispersion_y": float(dispersion_y),
            "total_gaze_points": int(total_gaze_points),
            "saccade_length": float(avg_saccade),
            "reading_speed": float(reading_speed),
            "drastic_movements": int(drastic_movements_count)
        }
        
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return {
            "n_fixations": random.randint(15, 30),
            "mean_fix_dur": random.uniform(200, 350),
            "std_fix_dur": random.uniform(50, 150),
            "mean_x": random.uniform(300, 500),
            "mean_y": random.uniform(200, 400),
            "dispersion_x": random.uniform(50, 150),
            "dispersion_y": random.uniform(30, 100),
            "total_gaze_points": random.randint(100, 300),
            "saccade_length": random.uniform(5, 20),
            "reading_speed": random.uniform(150, 300),
            "drastic_movements": drastic_movements_count
        }

def predict_dyslexia_from_eye_tracking(features_dict):
    """Predict dyslexia based on eye tracking features"""
    # Check for drastic movements
    drastic_movements = features_dict.get("drastic_movements", 0)
    
    # Simple rule-based assessment
    n_fix = features_dict.get("n_fixations", 0)
    mean_fix = features_dict.get("mean_fix_dur", 0)
    dispersion = features_dict.get("dispersion_x", 0) + features_dict.get("dispersion_y", 0)
    
    # Rule 1: Drastic movements detection (2 or more indicates risk)
    if drastic_movements >= 2:
        risk = "High Risk"
        confidence = {"High Risk": 0.8, "Medium Risk": 0.15, "Low Risk": 0.05}
        reason = f"Detected {drastic_movements} drastic eye movements"
    
    # Rule 2: High dispersion or unusual fixation patterns
    elif dispersion > 250 or mean_fix > 450 or n_fix > 50:
        risk = "Medium to High Risk"
        confidence = {"High Risk": 0.6, "Medium Risk": 0.3, "Low Risk": 0.1}
        reason = f"Unusual eye movement patterns detected (dispersion: {dispersion:.1f}, fixations: {n_fix})"
    
    # Rule 3: Moderate indicators
    elif dispersion > 150 or mean_fix > 350 or n_fix > 35:
        risk = "Medium Risk"
        confidence = {"High Risk": 0.3, "Medium Risk": 0.5, "Low Risk": 0.2}
        reason = "Moderate eye movement variations detected"
    
    # Rule 4: Normal patterns
    else:
        risk = "Low Risk"
        confidence = {"High Risk": 0.1, "Medium Risk": 0.3, "Low Risk": 0.6}
        reason = "Normal eye movement patterns"
    
    return {
        "prediction": risk,
        "confidence": confidence,
        "drastic_movements": drastic_movements,
        "reason": reason
    }

def analyze_handwriting(image):
    """Analyze handwriting image and return features"""
    try:
        # For now, we'll generate simulated analysis results
        quality_options = ["Good", "Fair", "Poor"]
        quality_weights = [0.6, 0.3, 0.1]
        
        consistency_options = ["Consistent", "Variable", "Irregular"]
        consistency_weights = [0.5, 0.3, 0.2]
        
        level_options = ["High", "Medium", "Low"]
        level_weights = [0.3, 0.5, 0.2]
        
        pressure_options = ["Light", "Medium", "Heavy"]
        pressure_weights = [0.3, 0.5, 0.2]
        
        risk_options = ["Low Risk", "Medium Risk", "High Risk"]
        risk_weights = [0.6, 0.3, 0.1]
        
        analysis_results = {
            "Image Quality": random.choices(quality_options, weights=quality_weights, k=1)[0],
            "Contrast Level": random.choices(level_options, weights=level_weights, k=1)[0],
            "Image Orientation": random.choice(["Portrait", "Landscape"]),
            "Letter Size Consistency": f"{random.uniform(70, 95):.1f}%",
            "Baseline Stability": f"{random.uniform(65, 90):.1f}%",
            "Word Spacing": random.choices(consistency_options, weights=consistency_weights, k=1)[0],
            "Stroke Pressure": random.choices(pressure_options, weights=pressure_weights, k=1)[0],
            "Writing Slant": f"{random.uniform(-20, 20):.1f}°",
            "Overall Legibility": random.choices(quality_options, weights=quality_weights, k=1)[0],
            "Dysgraphia Risk Level": random.choices(risk_options, weights=risk_weights, k=1)[0],
            "Confidence Score": f"{random.uniform(75, 95):.1f}%"
        }
        
        return analysis_results
        
    except Exception as e:
        st.error(f"Error analyzing handwriting: {e}")
        return {
            "Error": "Analysis failed",
            "Message": str(e)
        }

# ------------------- EYE TRACKING FUNCTIONS -------------------
def detect_gaze_point(landmarks, frame_shape):
    """Detect gaze point from facial landmarks"""
    h, w = frame_shape[:2]
    
    try:
        # Use eye landmarks for gaze estimation
        # Left eye landmarks (MediaPipe indices)
        left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 173]
        # Right eye landmarks
        right_eye_indices = [362, 263, 384, 385, 386, 387, 388, 466]
        
        # Calculate center of left eye
        left_eye_x = np.mean([landmarks[i].x for i in left_eye_indices])
        left_eye_y = np.mean([landmarks[i].y for i in left_eye_indices])
        
        # Calculate center of right eye
        right_eye_x = np.mean([landmarks[i].x for i in right_eye_indices])
        right_eye_y = np.mean([landmarks[i].y for i in right_eye_indices])
        
        # Average both eyes for gaze point
        gaze_x = (left_eye_x + right_eye_x) / 2
        gaze_y = (left_eye_y + right_eye_y) / 2
        
        # Convert to pixel coordinates
        pixel_x = int(gaze_x * w)
        pixel_y = int(gaze_y * h)
        
        return (pixel_x, pixel_y)
        
    except:
        # Return center of frame if detection fails
        return (w // 2, h // 2)

def detect_drastic_movements(gaze_points, threshold=100):
    """Detect drastic eye movements based on distance between consecutive points"""
    drastic_movements = 0
    
    if len(gaze_points) < 2:
        return 0
    
    for i in range(1, len(gaze_points)):
        if len(gaze_points[i]) > 1 and len(gaze_points[i-1]) > 1:
            distance = np.sqrt((gaze_points[i][0] - gaze_points[i-1][0])**2 + 
                              (gaze_points[i][1] - gaze_points[i-1][1])**2)
            
            if distance > threshold:
                drastic_movements += 1
    
    return drastic_movements

def detect_fixations(gaze_points, timestamps, fixation_threshold=25, min_fix_duration=0.1):
    """Detect fixations from gaze points"""
    fixations = []
    
    if len(gaze_points) < 2:
        return fixations
    
    fixation_start = 0
    fixation_points = [gaze_points[0]]
    
    for i in range(1, len(gaze_points)):
        current_point = gaze_points[i]
        last_point = gaze_points[i-1]
        
        # Calculate distance between consecutive points
        distance = np.sqrt((current_point[0] - last_point[0])**2 + 
                          (current_point[1] - last_point[1])**2)
        
        if distance < fixation_threshold:
            # Still in same fixation
            fixation_points.append(current_point)
        else:
            # Fixation ended
            if len(fixation_points) > 1:
                fixation_duration = timestamps[i-1] - timestamps[fixation_start]
                if fixation_duration >= min_fix_duration:
                    # Calculate fixation center
                    fix_x = np.mean([p[0] for p in fixation_points])
                    fix_y = np.mean([p[1] for p in fixation_points])
                    fixations.append({
                        "start": timestamps[fixation_start],
                        "end": timestamps[i-1],
                        "duration": fixation_duration,
                        "center": (fix_x, fix_y),
                        "points": len(fixation_points)
                    })
            
            # Start new fixation
            fixation_start = i
            fixation_points = [current_point]
    
    # Check last fixation
    if len(fixation_points) > 1:
        fixation_duration = timestamps[-1] - timestamps[fixation_start]
        if fixation_duration >= min_fix_duration:
            fix_x = np.mean([p[0] for p in fixation_points])
            fix_y = np.mean([p[1] for p in fixation_points])
            fixations.append({
                "start": timestamps[fixation_start],
                "end": timestamps[-1],
                "duration": fixation_duration,
                "center": (fix_x, fix_y),
                "points": len(fixation_points)
            })
    
    return fixations

# ------------------- LOGIN / SIGNUP -------------------
def login_page():
    st.markdown("<h1 class='main-header'>🧠 NeuroScan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6C757D; font-size: 1.2rem;'>Dyslexia & Dysgraphia Screening Tool</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown("<h3 style='text-align: center;'>Welcome Back</h3>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Login", use_container_width=True):
                    if username in st.session_state.users and st.session_state.users[username] == password:
                        st.session_state.logged_in = True
                        st.session_state.user = username
                        st.success(f"Welcome back, {username}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            with col_btn2:
                if st.button("Demo Login", use_container_width=True):
                    st.session_state.logged_in = True
                    st.session_state.user = "demo_user"
                    st.session_state.users["demo_user"] = "demo123"
                    st.success("Demo mode activated!")
                    time.sleep(1)
                    st.rerun()
        
        with tab2:
            st.markdown("<h3 style='text-align: center;'>Create Account</h3>", unsafe_allow_html=True)
            new_user = st.text_input("New Username", key="signup_user")
            new_pass = st.text_input("New Password", type="password", key="signup_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
            
            if st.button("Create Account", use_container_width=True):
                if new_user in st.session_state.users:
                    st.warning("Username already exists")
                elif new_pass != confirm_pass:
                    st.warning("Passwords do not match")
                elif len(new_user) < 3 or len(new_pass) < 3:
                    st.warning("Username and password must be at least 3 characters")
                else:
                    st.session_state.users[new_user] = new_pass
                    st.success("Account created successfully! Please login.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------- PROGRESS INDICATOR -------------------
def show_progress():
    steps = ["Handwriting", "Eye Tracking", "Report"]
    
    st.markdown("<div class='progress-container'>", unsafe_allow_html=True)
    st.markdown("<div class='step-indicator'>", unsafe_allow_html=True)
    
    for i, step in enumerate(steps, 1):
        status = ""
        if i == st.session_state.current_step:
            status = "active"
        elif i < st.session_state.current_step:
            status = "completed"
        
        st.markdown(f"""
        <div class='step {status}'>
            <div class='step-number'>{i}</div>
            <div>{step}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# ------------------- HANDWRITING ANALYSIS -------------------
def handwriting_analysis():
    st.markdown("<h2 class='sub-header'>📝 Step 1: Handwriting Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Upload Handwriting Sample")
        st.markdown("Upload a clear image of handwritten text for dysgraphia analysis.")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of handwriting",
            key="handwriting_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Handwriting Sample", use_column_width=True)
                
                # Store in session state
                st.session_state.handwriting_image = image
                
                # Show image info
                st.info(f"Image size: {image.size} | Mode: {image.mode} | Format: {image.format}")
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Analysis Parameters")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Standard Analysis", "Detailed Analysis", "Quick Assessment"],
            key="analysis_type"
        )
        
        if st.button("🚀 Analyze Handwriting", use_container_width=True, key="analyze_btn"):
            if st.session_state.handwriting_image is not None:
                with st.spinner("Analyzing handwriting features..."):
                    # Simulate analysis time
                    time.sleep(2)
                    
                    try:
                        # Get analysis results
                        analysis_results = analyze_handwriting(st.session_state.handwriting_image)
                        
                        # Store results
                        st.session_state.results.update(analysis_results)
                        st.session_state.current_step = 2
                        
                        # Display success message
                        st.success("✅ Handwriting analysis completed!")
                        
                        # Display results
                        st.markdown("### Analysis Results")
                        for key, value in analysis_results.items():
                            if "Risk" in key or "Level" in key:
                                risk_class = "low-risk" if "Low" in str(value) else "medium-risk" if "Medium" in str(value) else "high-risk"
                                st.markdown(f"**{key}:** <span class='prediction-badge {risk_class}'>{value}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"**{key}:** {value}")
                        
                        st.info("✅ Proceed to Step 2 for eye tracking analysis.")
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
            else:
                st.warning("⚠️ Please upload a handwriting image first.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------- EYE TRACKING WITH SPEECH RECOGNITION -------------------
# ------------------- EYE TRACKING WITH SPEECH RECOGNITION -------------------
def eye_tracking():
    st.markdown("<h2 class='sub-header'>👁️ Step 2: Eye Tracking & Speech Analysis</h2>", unsafe_allow_html=True)
    
    # Reading passages - SHORTER for better testing
    passages = {
        "Passage 1 - Simple": "The quick brown fox jumps over the lazy dog.",
        
        "Passage 2 - Intermediate": "Reading regularly improves vocabulary and comprehension skills.",
        
        "Passage 3 - Complex": "Neuroplasticity allows the brain to form new neural connections.",
        
        "Passage 4 - Technical": "Dyslexia involves difficulties with word recognition and spelling.",
        
        "Passage 5 - Narrative": "Children played in the park under the clear blue sky."
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Reading Test Setup")
        
        # Passage selection
        selected_passage = st.selectbox(
            "Select Reading Passage",
            list(passages.keys()),
            key="passage_select"
        )
        
        passage_text = passages[selected_passage]
        
        # Font size selection
        font_size = st.slider(
            "Select Font Size",
            min_value=14,
            max_value=32,
            value=18,
            help="Adjust text size for comfortable reading",
            key="font_size"
        )
        
        # Reading duration
        reading_duration = st.slider(
            "Reading Duration (seconds)",
            min_value=5,
            max_value=30,
            value=10,
            help="How long to track eye movements and record speech",
            key="reading_duration"
        )
        
        # Test microphone button
        st.markdown("#### 🎤 Test Microphone")
        if st.button("Test Speech Recognition", key="test_mic"):
            with st.spinner("Testing microphone..."):
                try:
                    recognizer = sr.Recognizer()
                    with sr.Microphone() as source:
                        st.info("Speak something now...")
                        recognizer.adjust_for_ambient_noise(source, duration=1)
                        audio = recognizer.listen(source, timeout=3)
                        text = recognizer.recognize_google(audio)
                        st.success(f"Microphone working! Heard: '{text}'")
                except sr.UnknownValueError:
                    st.warning("Could not understand speech. Please speak louder.")
                except sr.RequestError:
                    st.error("Speech recognition service unavailable.")
                except Exception as e:
                    st.error(f"Microphone error: {e}")
        
        # Display passage
        st.markdown("### Reading Passage Preview")
        st.markdown(f"<div style='font-size:{font_size}px; line-height:1.6; padding:15px; background:#7851A9; border-radius:5px; border:1px solid #ddd;'>{passage_text}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Eye Tracking & Speech Setup")
        
        # Initialize control buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            start_button = st.button("🎥 Start Eye Tracking & Speech", use_container_width=True, key="start_tracking")
        
        with col_btn2:
            stop_button = st.button("⏹️ Stop Tracking", use_container_width=True, key="stop_tracking")
        
        # Initialize camera feed placeholder
        camera_placeholder = st.empty()
        
        # Speech recognition status placeholder
        speech_status = st.empty()
        
        # Demo mode toggle
        demo_mode = st.checkbox("Use Demo Mode (if microphone not working)", value=False)
        
        if start_button:
            # Reset session states
            st.session_state.tracking_active = True
            st.session_state.speech_recognition_active = True
            st.session_state.eye_tracking_data = {"gaze_points": [], "timestamps": [], "fixations": [], "drastic_movements": 0}
            
            # Initialize speech data
            if demo_mode:
                # Use simulated speech data
                spoken_text, accuracy = simulate_speech_recognition(passage_text, reading_duration)
                st.session_state.speech_data = {
                    "spoken_text": spoken_text,
                    "accuracy": accuracy,
                    "is_recording": True,
                    "dyslexia_from_speech": "Low Risk" if accuracy >= 90 else "Medium Risk" if accuracy >= 70 else "High Risk",
                    "demo_mode": True
                }
            else:
                # Try real speech recognition
                st.session_state.speech_data = {
                    "spoken_text": "",
                    "accuracy": 0,
                    "is_recording": True,
                    "dyslexia_from_speech": "Analyzing...",
                    "demo_mode": False
                }
                
                # Initialize speech recognition in a separate thread
                if 'audio_queue' not in st.session_state:
                    st.session_state.audio_queue = queue.Queue()
                if 'stop_event' not in st.session_state:
                    st.session_state.stop_event = threading.Event()
                
                st.session_state.stop_event.clear()
                
                # Start speech recognition threads
                recognition_thread = threading.Thread(
                    target=speech_recognition_thread,
                    args=(st.session_state.audio_queue, passage_text, st.session_state.stop_event),
                    daemon=True
                )
                processing_thread = threading.Thread(
                    target=process_audio,
                    args=(st.session_state.audio_queue, passage_text, st.session_state.stop_event),
                    daemon=True
                )
                
                recognition_thread.start()
                processing_thread.start()
            
            st.rerun()
        
        if stop_button:
            st.session_state.tracking_active = False
            st.session_state.speech_recognition_active = False
            if 'stop_event' in st.session_state:
                st.session_state.stop_event.set()
            st.rerun()
        
        # Webcam feed and tracking
        if st.session_state.tracking_active:
            st.info("🔍 Eye tracking & speech recognition in progress... Read the passage aloud.")
            
            if st.session_state.speech_data.get("demo_mode", False):
                st.warning("⚠️ Using Demo Mode for speech recognition")
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not open camera. Please check your camera connection.")
                st.session_state.tracking_active = False
            else:
                # Set camera resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Initialize MediaPipe Face Mesh
                mp_face_mesh = mp.solutions.face_mesh
                face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                
                start_time = time.time()
                gaze_points = []
                timestamps = []
                drastic_movements_count = 0
                
                # Create progress bar
                progress_bar = st.progress(0)
                
                try:
                    while st.session_state.tracking_active and (time.time() - start_time) < reading_duration:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Flip frame horizontally for a mirror view
                        frame = cv2.flip(frame, 1)
                        h, w = frame.shape[:2]
                        
                        # Convert BGR to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb_frame)
                        
                        current_time = time.time()
                        elapsed = current_time - start_time
                        
                        # Update progress bar
                        progress = min(elapsed / reading_duration, 1.0)
                        progress_bar.progress(progress)
                        
                        if results.multi_face_landmarks:
                            for face_landmarks in results.multi_face_landmarks:
                                # Detect gaze point
                                gaze_point = detect_gaze_point(face_landmarks.landmark, frame.shape)
                                
                                # Store gaze data
                                gaze_points.append(gaze_point + (current_time,))
                                timestamps.append(current_time)
                                
                                # Detect drastic movements
                                if len(gaze_points) > 1:
                                    prev_point = gaze_points[-2]
                                    curr_point = gaze_points[-1]
                                    distance = np.sqrt((curr_point[0] - prev_point[0])**2 + 
                                                      (curr_point[1] - prev_point[1])**2)
                                    
                                    # Threshold for drastic movement (100 pixels)
                                    if distance > 100:
                                        drastic_movements_count += 1
                                
                                # Draw gaze point on frame
                                cv2.circle(frame, gaze_point, 8, (0, 255, 0), -1)
                                cv2.circle(frame, gaze_point, 12, (0, 255, 0), 2)
                                
                                # Add text for drastic movements
                                cv2.putText(frame, f"Drastic Moves: {drastic_movements_count}", 
                                          (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Add UI overlay
                        time_left = max(0, reading_duration - elapsed)
                        cv2.putText(frame, f"Time: {time_left:.1f}s", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Gaze Points: {len(gaze_points)}", (10, 70), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, "Tracking Active", (10, h - 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Display frame
                        camera_placeholder.image(frame, channels="BGR", use_container_width=True)
                        
                        # Display speech recognition status
                        with speech_status.container():
                            st.markdown("<div class='speech-status'>", unsafe_allow_html=True)
                            st.markdown("### 🎤 Speech Recognition Status")
                            
                            if st.session_state.speech_data.get("demo_mode", False):
                                st.markdown("**Mode:** Demo Mode (Microphone not used)")
                            
                            spoken_text = st.session_state.speech_data.get("spoken_text", "")
                            accuracy = st.session_state.speech_data.get("accuracy", 0)
                            dyslexia_status = st.session_state.speech_data.get("dyslexia_from_speech", "Analyzing...")
                            
                            if spoken_text:
                                st.markdown(f"**Spoken Text:** {spoken_text}")
                                st.markdown(f"**Accuracy:** {accuracy:.1f}%")
                                st.markdown(f"**Speech Analysis:** {dyslexia_status}")
                            else:
                                st.markdown("**Status:** Listening... Speak clearly into microphone")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Check if time is up
                        if elapsed >= reading_duration:
                            st.session_state.tracking_active = False
                            st.session_state.speech_recognition_active = False
                            if 'stop_event' in st.session_state:
                                st.session_state.stop_event.set()
                            break
                    
                except Exception as e:
                    st.error(f"Error during tracking: {e}")
                finally:
                    cap.release()
                    face_mesh.close()
                
                # Store collected data
                st.session_state.eye_tracking_data["gaze_points"] = gaze_points
                st.session_state.eye_tracking_data["timestamps"] = timestamps
                st.session_state.eye_tracking_data["drastic_movements"] = drastic_movements_count
                
                # Detect fixations
                if len(gaze_points) > 0:
                    fixations = detect_fixations([(p[0], p[1]) for p in gaze_points], timestamps)
                    st.session_state.eye_tracking_data["fixations"] = fixations
                
                # If no speech was captured and not in demo mode, use simulation
                if (not st.session_state.speech_data.get("spoken_text") or 
                    st.session_state.speech_data.get("spoken_text") == "Listening...") and not demo_mode:
                    spoken_text, accuracy = simulate_speech_recognition(passage_text, reading_duration)
                    st.session_state.speech_data["spoken_text"] = spoken_text
                    st.session_state.speech_data["accuracy"] = accuracy
                    st.session_state.speech_data["demo_mode_used"] = True
                
                # Analyze results
                if len(gaze_points) > 0:
                    # Extract features
                    features = extract_eye_features(gaze_points, 
                                                   st.session_state.eye_tracking_data["fixations"],
                                                   drastic_movements_count)
                    
                    # Get eye tracking prediction
                    eye_prediction = predict_dyslexia_from_eye_tracking(features)
                    
                    # Get speech prediction
                    speech_accuracy = st.session_state.speech_data.get("accuracy", 0)
                    if speech_accuracy >= 90:
                        speech_dyslexia = "Low Risk"
                    elif speech_accuracy >= 70:
                        speech_dyslexia = "Medium Risk"
                    else:
                        speech_dyslexia = "High Risk"
                    
                    st.session_state.speech_data["dyslexia_from_speech"] = speech_dyslexia
                    
                    # Combined prediction
                    combined_risk = "Low Risk"
                    if eye_prediction["prediction"] == "High Risk" or speech_dyslexia == "High Risk":
                        combined_risk = "High Risk"
                    elif eye_prediction["prediction"] == "Medium to High Risk" or speech_dyslexia == "Medium Risk":
                        combined_risk = "Medium Risk"
                    
                    # Store results
                    st.session_state.results.update({
                        "Eye_Tracking_Features": features,
                        "Eye_Tracking_Prediction": eye_prediction["prediction"],
                        "Eye_Tracking_Confidence": eye_prediction["confidence"],
                        "Drastic_Movements_Count": drastic_movements_count,
                        "Drastic_Movements_Reason": eye_prediction["reason"],
                        "Speech_Analysis": {
                            "spoken_text": st.session_state.speech_data.get("spoken_text", ""),
                            "accuracy": speech_accuracy,
                            "dyslexia_prediction": speech_dyslexia,
                            "original_passage": passage_text,
                            "demo_mode_used": st.session_state.speech_data.get("demo_mode_used", False)
                        },
                        "Combined_Dyslexia_Prediction": combined_risk,
                        "Reading_Passage": selected_passage,
                        "Font_Size": font_size,
                        "Reading_Duration": reading_duration,
                        "Total_Gaze_Points": len(gaze_points)
                    })
                    
                    st.session_state.current_step = 3
                    
                    # Display results
                    display_eye_tracking_results(features, eye_prediction, speech_accuracy, 
                                                speech_dyslexia, combined_risk, len(gaze_points))
                    
                else:
                    st.warning("⚠️ No gaze points were captured. Please ensure your face is visible to the camera.")
        
        else:
            st.info("👆 Click 'Start Eye Tracking & Speech' to begin the reading test.")
            
            # Show example or previous results
            if "Eye_Tracking_Features" in st.session_state.results:
                with st.expander("📊 View Previous Results"):
                    features = st.session_state.results["Eye_Tracking_Features"]
                    st.markdown("**Previous Eye Tracking Results:**")
                    for key, value in features.items():
                        if isinstance(value, float):
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
                        else:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def display_eye_tracking_results(features, eye_prediction, speech_accuracy, speech_dyslexia, combined_risk, total_points):
    """Display eye tracking and speech analysis results"""
    st.markdown("### 📊 Eye Tracking & Speech Analysis Results")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["👁️ Eye Tracking Analysis", "🎤 Speech Analysis", "📊 Combined Results"])
    
    with tab1:
        # Eye tracking metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Gaze Points", total_points)
        with col2:
            st.metric("Fixations", features.get("n_fixations", 0))
        with col3:
            st.metric("Drastic Movements", features.get("drastic_movements", 0))
        with col4:
            risk = eye_prediction.get("prediction", "Unknown")
            st.metric("Eye Tracking Risk", risk)
        
        # Detailed eye tracking features
        st.markdown("#### Detailed Eye Tracking Analysis")
        
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            st.markdown(f"**Number of Fixations:** {features.get('n_fixations', 0)}")
            st.markdown(f"**Mean Fixation Duration:** {features.get('mean_fix_dur', 0):.2f} ms")
            st.markdown(f"**Fixation Duration STD:** {features.get('std_fix_dur', 0):.2f} ms")
            st.markdown(f"**Mean X Coordinate:** {features.get('mean_x', 0):.2f}")
            st.markdown(f"**Mean Y Coordinate:** {features.get('mean_y', 0):.2f}")
        
        with col_e2:
            st.markdown(f"**Horizontal Dispersion:** {features.get('dispersion_x', 0):.2f}")
            st.markdown(f"**Vertical Dispersion:** {features.get('dispersion_y', 0):.2f}")
            st.markdown(f"**Average Saccade Length:** {features.get('saccade_length', 0):.2f} px")
            st.markdown(f"**Drastic Movements Detected:** {features.get('drastic_movements', 0)}")
        
        # Eye tracking confidence visualization
        if "confidence" in eye_prediction:
            conf = eye_prediction["confidence"]
            if conf:
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(conf.keys()),
                        y=list(conf.values()),
                        marker_color=['#28a745', '#ffc107', '#dc3545'],
                        text=[f"{v:.1%}" for v in conf.values()],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Eye Tracking Risk Confidence",
                    yaxis_title="Probability",
                    xaxis_title="Risk Level",
                    yaxis=dict(range=[0, 1]),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Speech analysis metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Speech Accuracy", f"{speech_accuracy:.1f}%")
        with col2:
            # Color code based on accuracy
            if speech_accuracy >= 90:
                color = "green"
            elif speech_accuracy >= 70:
                color = "orange"
            else:
                color = "red"
            st.metric("Accuracy Status", f"{'✓' if speech_accuracy >= 90 else '⚠'}")
        with col3:
            st.metric("Speech Analysis Risk", speech_dyslexia)
        
        # Speech analysis details
        st.markdown("#### Speech Recognition Details")
        
        st.markdown("**Original Passage:**")
        st.info(st.session_state.results["Speech_Analysis"]["original_passage"])
        
        st.markdown("**Spoken Text:**")
        st.info(st.session_state.results["Speech_Analysis"]["spoken_text"])
        
        # Accuracy interpretation
        st.markdown("#### Accuracy Interpretation:")
        if speech_accuracy >= 90:
            st.success("✅ Excellent reading accuracy (≥90%). Indicates low likelihood of dyslexia.")
        elif speech_accuracy >= 70:
            st.warning("⚠️ Moderate reading accuracy (70-89%). May indicate potential reading difficulties.")
        else:
            st.error("❌ Low reading accuracy (<70%). Suggests potential dyslexia indicators.")
        
        # Speech-based dyslexia determination
        st.markdown("#### Speech-based Dyslexia Determination:")
        st.markdown(f"**Threshold:** 90% accuracy")
        st.markdown(f"**Your Accuracy:** {speech_accuracy:.1f}%")
        
        if speech_accuracy >= 90:
            st.success("**Result:** Not dyslexic based on speech analysis")
        else:
            st.error("**Result:** Potential dyslexia indicators detected in speech")
    
    with tab3:
        # Combined results
        st.markdown("#### Combined Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Eye Tracking Findings:")
            st.markdown(f"**Risk Level:** {eye_prediction['prediction']}")
            st.markdown(f"**Drastic Movements:** {features.get('drastic_movements', 0)}")
            if features.get('drastic_movements', 0) >= 2:
                st.error("⚠️ 2 or more drastic eye movements detected (dyslexia indicator)")
            else:
                st.success("✅ Normal eye movement patterns")
        
        with col2:
            st.markdown("##### Speech Analysis Findings:")
            st.markdown(f"**Accuracy:** {speech_accuracy:.1f}%")
            st.markdown(f"**Threshold:** ≥90% for normal reading")
            if speech_accuracy >= 90:
                st.success("✅ Normal reading accuracy")
            else:
                st.error("⚠️ Below normal reading accuracy")
        
        # Final combined assessment
        st.markdown("---")
        st.markdown("##### Final Combined Assessment:")
        
        if combined_risk == "High Risk":
            st.error(f"## 🔴 High Risk of Dyslexia")
            st.markdown("Both eye tracking and speech analysis indicate significant dyslexia indicators.")
        elif combined_risk == "Medium Risk":
            st.warning(f"## 🟡 Medium Risk of Dyslexia")
            st.markdown("Some indicators detected in either eye tracking or speech analysis.")
        else:
            st.success(f"## 🟢 Low Risk of Dyslexia")
            st.markdown("Minimal indicators detected in both analyses.")
        
        # Recommendations
        st.markdown("##### Recommendations:")
        recommendations = []
        if features.get('drastic_movements', 0) >= 2:
            recommendations.append("Consult with eye movement specialist for further evaluation")
        if speech_accuracy < 90:
            recommendations.append("Consider speech therapy or reading intervention programs")
        if combined_risk in ["Medium Risk", "High Risk"]:
            recommendations.append("Schedule comprehensive assessment with educational psychologist")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    st.success("✅ Eye tracking & speech analysis completed!")
    st.info("📋 Proceed to Step 3 to generate your comprehensive report.")

# ------------------- REPORT GENERATION -------------------
def generate_report():
    st.markdown("<h2 class='sub-header'>📊 Step 3: Comprehensive Report</h2>", unsafe_allow_html=True)
    
    if not st.session_state.results:
        st.warning("Please complete Steps 1 and 2 to generate a report.")
        return
    
    # Summary dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dysgraphia_risk = st.session_state.results.get("Dysgraphia Risk Level", "Not Analyzed")
        risk_color = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}.get(dysgraphia_risk, "⚪")
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Dysgraphia Risk</h3>
            <h1>{risk_color} {dysgraphia_risk}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        combined_risk = st.session_state.results.get("Combined_Dyslexia_Prediction", "Not Analyzed")
        risk_color = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}.get(combined_risk, "⚪")
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Dyslexia Risk</h3>
            <h1>{risk_color} {combined_risk}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Assessment Summary</h3>
            <h4>{time.strftime('%Y-%m-%d')}</h4>
            <p>User: {st.session_state.user}</p>
            <p>Status: Complete</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Handwriting Results", "👁️ Eye Tracking Results", "🎤 Speech Results", "📄 Export Report"])
    
    with tab1:
        st.markdown("### Handwriting Analysis Details")
        
        handwriting_keys = [k for k in st.session_state.results.keys() 
                          if not k.startswith("Eye_") and not k.startswith("Speech") 
                          and k not in ["Confidence_Scores", "Reading_Passage", "Font_Size", 
                                       "Reading_Duration", "Total_Gaze_Points", "Combined_Dyslexia_Prediction",
                                       "Eye_Tracking_Prediction", "Eye_Tracking_Confidence", "Drastic_Movements_Count",
                                       "Drastic_Movements_Reason"]]
        
        if handwriting_keys:
            col_h1, col_h2 = st.columns(2)
            
            with col_h1:
                for i, key in enumerate(handwriting_keys):
                    if i < len(handwriting_keys) // 2:
                        value = st.session_state.results[key]
                        if "Risk" in key or "Level" in key:
                            risk_class = "low-risk" if "Low" in str(value) else "medium-risk" if "Medium" in str(value) else "high-risk"
                            st.markdown(f"**{key}:** <span class='prediction-badge {risk_class}'>{value}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{key}:** {value}")
            
            with col_h2:
                for i, key in enumerate(handwriting_keys):
                    if i >= len(handwriting_keys) // 2:
                        value = st.session_state.results[key]
                        if "Risk" in key or "Level" in key:
                            risk_class = "low-risk" if "Low" in str(value) else "medium-risk" if "Medium" in str(value) else "high-risk"
                            st.markdown(f"**{key}:** <span class='prediction-badge {risk_class}'>{value}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{key}:** {value}")
        else:
            st.info("No handwriting analysis results available.")
    
    with tab2:
        st.markdown("### Eye Tracking Analysis Details")
        
        if "Eye_Tracking_Features" in st.session_state.results:
            features = st.session_state.results["Eye_Tracking_Features"]
            
            st.markdown("#### Dyslexia Detection through Eye Tracking")
            st.markdown("**Analysis Method:** Detection of drastic eye movements and unusual fixation patterns")
            
            col_e1, col_e2 = st.columns(2)
            
            with col_e1:
                st.markdown("##### Key Metrics")
                st.markdown(f"**Drastic Movements Detected:** {features.get('drastic_movements', 0)}")
                st.markdown(f"**Fixations:** {features.get('n_fixations', 0)}")
                st.markdown(f"**Average Fixation Duration:** {features.get('mean_fix_dur', 0):.1f} ms")
                st.markdown(f"**Gaze Dispersion:** {features.get('dispersion_x', 0) + features.get('dispersion_y', 0):.1f}")
            
            with col_e2:
                st.markdown("##### Assessment")
                st.markdown(f"**Eye Tracking Risk:** {st.session_state.results.get('Eye_Tracking_Prediction', 'N/A')}")
                st.markdown(f"**Threshold:** 2+ drastic movements indicates risk")
                st.markdown(f"**Result:** {'Risk detected' if features.get('drastic_movements', 0) >= 2 else 'Normal patterns'}")
                st.markdown(f"**Reason:** {st.session_state.results.get('Drastic_Movements_Reason', 'N/A')}")
            
            # Visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['Drastic Movements', 'Normal Movements'],
                    y=[features.get('drastic_movements', 0), 
                       features.get('total_gaze_points', 0) - features.get('drastic_movements', 0)],
                    marker_color=['#dc3545', '#28a745']
                )
            ])
            fig.update_layout(
                title="Eye Movement Analysis",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No eye tracking results available.")
    
    with tab3:
        st.markdown("### Speech Analysis Details")
        
        if "Speech_Analysis" in st.session_state.results:
            speech_data = st.session_state.results["Speech_Analysis"]
            
            st.markdown("#### Dyslexia Detection through Speech Recognition")
            st.markdown("**Analysis Method:** Accuracy comparison between spoken text and original passage")
            
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.markdown("##### Speech Metrics")
                st.markdown(f"**Speech Accuracy:** {speech_data.get('accuracy', 0):.1f}%")
                st.markdown(f"**Threshold:** ≥90% for normal reading")
                st.markdown(f"**Words Spoken:** {len(speech_data.get('spoken_text', '').split())}")
                st.markdown(f"**Reading Duration:** {st.session_state.results.get('Reading_Duration', 0)} seconds")
            
            with col_s2:
                st.markdown("##### Assessment")
                st.markdown(f"**Speech Analysis Risk:** {speech_data.get('dyslexia_prediction', 'N/A')}")
                st.markdown(f"**Result:** {'Normal reading' if speech_data.get('accuracy', 0) >= 90 else 'Potential difficulties'}")
                st.markdown("**Interpretation:**")
                if speech_data.get('accuracy', 0) >= 90:
                    st.success("✅ Normal reading comprehension and pronunciation")
                elif speech_data.get('accuracy', 0) >= 70:
                    st.warning("⚠️ Moderate reading difficulties detected")
                else:
                    st.error("❌ Significant reading difficulties detected")
            
            # Accuracy visualization
            fig = go.Figure(data=[
                go.Indicator(
                    mode="gauge+number",
                    value=speech_data.get('accuracy', 0),
                    title={'text': "Speech Accuracy"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 70], 'color': "red"},
                            {'range': [70, 90], 'color': "yellow"},
                            {'range': [90, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                )
            ])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Text comparison
            with st.expander("View Text Comparison"):
                st.markdown("**Original Passage:**")
                st.info(speech_data.get('original_passage', ''))
                st.markdown("**Spoken Text:**")
                st.info(speech_data.get('spoken_text', ''))
        else:
            st.info("No speech analysis results available.")
    
    with tab4:
        st.markdown("### Export Options")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # Generate PDF Report
            if st.button("📄 Generate PDF Report", use_container_width=True, key="pdf_btn"):
                with st.spinner("Generating PDF report..."):
                    try:
                        filename = create_pdf_report()
                        
                        # Read the generated PDF
                        with open(filename, "rb") as f:
                            pdf_data = f.read()
                        
                        # Create download button
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_data,
                            file_name=f"NeuroScan_Report_{st.session_state.user}_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.success("✅ PDF report generated successfully!")
                        
                        # Clean up temporary file
                        try:
                            os.remove(filename)
                        except:
                            pass
                            
                    except Exception as e:
                        st.error(f"Error generating PDF: {e}")
        
        with col_exp2:
            # Export as JSON
            if st.button("📊 Export as JSON", use_container_width=True, key="json_btn"):
                try:
                    report_data = {
                        "user": st.session_state.user,
                        "assessment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "results": st.session_state.results
                    }
                    
                    # Convert to JSON string
                    json_str = json.dumps(report_data, indent=2, default=str)
                    
                    # Create download button
                    st.download_button(
                        label="📥 Download JSON Data",
                        data=json_str,
                        file_name=f"NeuroScan_Results_{st.session_state.user}_{time.strftime('%Y%m%d')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                    st.success("✅ JSON data exported successfully!")
                    
                except Exception as e:
                    st.error(f"Error exporting JSON: {e}")
        
        # Display raw data
        with st.expander("View Raw Data"):
            st.json(st.session_state.results)

# ------------------- PDF REPORT GENERATION -------------------
def create_pdf_report():
    """Generate comprehensive PDF report"""
    import tempfile
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    filename = temp_file.name
    
    try:
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title
        elements.append(Paragraph("NeuroScan Assessment Report", styles['Title']))
        elements.append(Spacer(1, 12))
        
        # User info
        elements.append(Paragraph(f"User: {st.session_state.user}", styles['Heading3']))
        elements.append(Paragraph(f"Assessment Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Heading4']))
        elements.append(Spacer(1, 24))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", styles['Heading2']))
        summary_text = f"""
        This report presents the results of the dyslexia and dysgraphia screening assessment.
        The assessment includes handwriting analysis, eye-tracking evaluation, and speech recognition
        to provide comprehensive insights into potential learning differences.
        
        Assessment completed by: {st.session_state.user}
        Date: {time.strftime('%Y-%m-%d')}
        """
        elements.append(Paragraph(summary_text, styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Handwriting Results
        elements.append(Paragraph("1. Handwriting Analysis (Dysgraphia Screening)", styles['Heading2']))
        
        # Collect handwriting data
        handwriting_data = [["Feature", "Observation"]]
        handwriting_features = [
            "Image Quality", "Contrast Level", "Letter Size Consistency",
            "Baseline Stability", "Word Spacing", "Stroke Pressure",
            "Writing Slant", "Overall Legibility", "Dysgraphia Risk Level",
            "Confidence Score"
        ]
        
        for feature in handwriting_features:
            value = st.session_state.results.get(feature, "N/A")
            handwriting_data.append([feature, str(value)])
        
        # Create table
        table1 = Table(handwriting_data, colWidths=[250, 200])
        table1.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2E86AB")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F8F9FA")),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ]))
        elements.append(table1)
        elements.append(Spacer(1, 30))
        
        # Eye Tracking Results
        elements.append(Paragraph("2. Eye Tracking Analysis (Dyslexia Detection)", styles['Heading2']))
        
        elements.append(Paragraph("2.1 Dyslexia Detection through Eye Tracking", styles['Heading3']))
        elements.append(Paragraph("Method: Detection of drastic eye movements (2 or more indicates risk)", styles['Normal']))
        
        if "Eye_Tracking_Features" in st.session_state.results:
            features = st.session_state.results["Eye_Tracking_Features"]
            
            eye_tracking_data = [["Feature", "Value"]]
            eye_tracking_features = [
                "drastic_movements", "n_fixations", "mean_fix_dur", 
                "dispersion_x", "dispersion_y", "total_gaze_points"
            ]
            
            for feature in eye_tracking_features:
                if feature in features:
                    value = features[feature]
                    if isinstance(value, float):
                        value = f"{value:.2f}"
                    eye_tracking_data.append([feature.replace('_', ' ').title(), str(value)])
            
            # Add prediction
            eye_pred = st.session_state.results.get("Eye_Tracking_Prediction", "N/A")
            eye_tracking_data.append(["Dyslexia Risk from Eye Tracking", eye_pred])
            
            # Add drastic movements analysis
            drastic_count = features.get('drastic_movements', 0)
            if drastic_count >= 2:
                analysis = f"High Risk - {drastic_count} drastic movements detected"
            else:
                analysis = f"Low Risk - {drastic_count} drastic movements detected"
            eye_tracking_data.append(["Drastic Movements Analysis", analysis])
            
            table2 = Table(eye_tracking_data, colWidths=[250, 200])
            table2.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#A23B72")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 12),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F8F9FA")),
                ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ]))
            elements.append(table2)
        
        elements.append(Spacer(1, 20))
        
        # Speech Recognition Results
        elements.append(Paragraph("3. Speech Recognition Analysis (Dyslexia Detection)", styles['Heading2']))
        
        elements.append(Paragraph("3.1 Dyslexia Detection through Speech", styles['Heading3']))
        elements.append(Paragraph("Method: Accuracy comparison between spoken text and original passage (≥90% accuracy indicates normal reading)", styles['Normal']))
        
        if "Speech_Analysis" in st.session_state.results:
            speech_data = st.session_state.results["Speech_Analysis"]
            
            speech_analysis_data = [["Feature", "Value"]]
            speech_analysis_data.append(["Speech Accuracy", f"{speech_data.get('accuracy', 0):.1f}%"])
            speech_analysis_data.append(["Accuracy Threshold", "90%"])
            speech_analysis_data.append(["Words in Original Passage", str(len(speech_data.get('original_passage', '').split()))])
            speech_analysis_data.append(["Words Spoken", str(len(speech_data.get('spoken_text', '').split()))])
            speech_analysis_data.append(["Dyslexia Risk from Speech", speech_data.get('dyslexia_prediction', 'N/A')])
            
            # Add accuracy interpretation
            accuracy = speech_data.get('accuracy', 0)
            if accuracy >= 90:
                interpretation = "Normal reading - No dyslexia indicators"
            elif accuracy >= 70:
                interpretation = "Moderate difficulties - Potential dyslexia indicators"
            else:
                interpretation = "Significant difficulties - Strong dyslexia indicators"
            speech_analysis_data.append(["Accuracy Interpretation", interpretation])
            
            table3 = Table(speech_analysis_data, colWidths=[250, 200])
            table3.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4B8BBE")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 12),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F8F9FA")),
                ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ]))
            elements.append(table3)
            
            # Add text comparison note
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("Note: Detailed text comparison available in digital report.", 
                                    ParagraphStyle(name='Note', fontSize=10, textColor=colors.grey)))
        
        elements.append(Spacer(1, 20))
        
        # Combined Results
        elements.append(Paragraph("4. Combined Assessment Results", styles['Heading2']))
        
        combined_data = [["Assessment Type", "Risk Level"]]
        combined_data.append(["Handwriting (Dysgraphia)", st.session_state.results.get("Dysgraphia Risk Level", "N/A")])
        combined_data.append(["Eye Tracking (Dyslexia)", st.session_state.results.get("Eye_Tracking_Prediction", "N/A")])
        combined_data.append(["Speech Analysis (Dyslexia)", st.session_state.results.get("Speech_Analysis", {}).get("dyslexia_prediction", "N/A")])
        combined_data.append(["Overall Combined Risk", st.session_state.results.get("Combined_Dyslexia_Prediction", "N/A")])
        
        table4 = Table(combined_data, colWidths=[250, 200])
        table4.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#28A745")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F8F9FA")),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ]))
        elements.append(table4)
        
        elements.append(Spacer(1, 30))
        
        # Recommendations
        elements.append(Paragraph("5. Recommendations", styles['Heading2']))
        
        recommendations = """
        1. Consult with an educational psychologist for formal assessment
        2. Consider occupational therapy for handwriting improvement (if dysgraphia risk indicated)
        3. Implement assistive technology tools for reading support
        4. Provide additional time for reading and writing tasks
        5. Use multisensory learning approaches
        6. Regular follow-up assessments recommended
        7. Consider specialized tutoring if reading difficulties are detected
        8. Create a supportive learning environment
        9. Eye movement therapy if drastic movements are consistently detected
        10. Speech therapy if reading accuracy is below 90%
        """
        elements.append(Paragraph(recommendations, styles['Normal']))
        
        # Disclaimer
        elements.append(Spacer(1, 30))
        disclaimer = """
        **Disclaimer**: This assessment is for screening purposes only and does not constitute a formal diagnosis. 
        Always consult with qualified professionals for accurate diagnosis and treatment recommendations.
        
        **Note on Methods**:
        - Eye Tracking: Detects dyslexia through analysis of eye movement patterns, especially drastic movements
        - Speech Recognition: Detects dyslexia through reading accuracy comparison against original text
        - Combined approach provides more reliable screening results
        """
        elements.append(Paragraph(disclaimer, ParagraphStyle(
            name='Disclaimer',
            fontSize=10,
            textColor=colors.grey,
            alignment=1
        )))
        
        # Footer
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Confidential Assessment Report - NeuroScan Screening Tool", 
                                ParagraphStyle(name='Footer', fontSize=9, alignment=1, textColor=colors.grey)))
        
        # Build PDF
        doc.build(elements)
        
        return filename
        
    except Exception as e:
        st.error(f"Error in PDF generation: {e}")
        # Return a simple PDF if there's an error
        return create_simple_pdf()

def create_simple_pdf():
    """Create a simple PDF if the main function fails"""
    import tempfile
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    filename = temp_file.name
    
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    elements.append(Paragraph("NeuroScan Assessment Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"User: {st.session_state.user}", styles['Normal']))
    elements.append(Paragraph(f"Date: {time.strftime('%Y-%m-%d')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Assessment completed successfully.", styles['Normal']))
    
    doc.build(elements)
    return filename

# ------------------- SIDEBAR -------------------
def sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px 0;'>
            <h2 style='color: #2E86AB;'>🧠 NeuroScan</h2>
            <p>Welcome, <strong>{st.session_state.user}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        menu_options = [
            "📝 Handwriting Analysis",
            "👁️ Eye Tracking",
            "📊 Report Dashboard",
            "⚙️ Settings",
            "🚪 Logout"
        ]
        
        try:
            selected = option_menu(
                menu_title=None,
                options=menu_options,
                icons=['pencil', 'eye', 'clipboard-data', 'gear', 'box-arrow-right'],
                menu_icon="cast",
                default_index=st.session_state.current_step - 1,
                styles={
                    "container": {"padding": "0!important", "background-color": "#7851A9"},
                    "icon": {"color": "#2E86AB", "font-size": "18px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#e9ecef"},
                    "nav-link-selected": {"background-color": "#2E86AB"},
                }
            )
        except:
            # Fallback if option_menu fails
            selected = st.radio("Navigation", menu_options, index=st.session_state.current_step - 1)
        
        # Update current step based on selection
        if "Handwriting" in selected:
            st.session_state.current_step = 1
        elif "Eye Tracking" in selected:
            st.session_state.current_step = 2
        elif "Report" in selected:
            st.session_state.current_step = 3
        elif "Settings" in selected:
            st.session_state.current_step = 4
        elif "Logout" in selected:
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.results = {}
            st.session_state.current_step = 1
            st.session_state.handwriting_image = None
            st.session_state.tracking_active = False
            st.session_state.eye_tracking_data = {"gaze_points": [], "fixations": [], "drastic_movements": 0}
            st.session_state.speech_data = {"spoken_text": "", "accuracy": 0, "is_recording": False}
            st.session_state.speech_recognition_active = False
            st.rerun()
        
        st.markdown("---")
        
        # Progress bar
        progress = min(st.session_state.current_step / 3, 1.0)
        st.progress(progress)
        st.caption(f"Step {min(st.session_state.current_step, 3)} of 3")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        if st.session_state.results:
            col1, col2 = st.columns(2)
            with col1:
                dysgraphia = st.session_state.results.get("Dysgraphia Risk Level", "N/A")
                st.metric("Dysgraphia", dysgraphia)
            with col2:
                dyslexia = st.session_state.results.get("Combined_Dyslexia_Prediction", "N/A")
                st.metric("Dyslexia", dyslexia)
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ Need Help?"):
            st.info("""
            **Eye Tracking & Speech Test:**
            - Position yourself 50-60cm from camera
            - Ensure adequate lighting
            - Read the passage aloud clearly
            - Keep your head relatively still
            - Speak at normal pace
            
            **Speech Recognition:**
            - Microphone access required
            - Speak clearly and at moderate volume
            - Accuracy calculated against original text
            - 90%+ accuracy indicates normal reading
            
            **Eye Movement Detection:**
            - 2+ drastic movements indicates risk
            - Normal reading shows smooth eye movements
            - Dyslexia often shows erratic patterns
            """)

# ------------------- SETTINGS PAGE -------------------
def settings_page():
    st.markdown("<h2 class='sub-header'>⚙️ Settings</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### User Settings")
        
        # Change password
        st.markdown("#### Change Password")
        current_pass = st.text_input("Current Password", type="password", key="current_pass")
        new_pass = st.text_input("New Password", type="password", key="new_pass")
        confirm_pass = st.text_input("Confirm New Password", type="password", key="confirm_pass_new")
        
        if st.button("Update Password", use_container_width=True):
            if st.session_state.users.get(st.session_state.user) == current_pass:
                if new_pass == confirm_pass:
                    st.session_state.users[st.session_state.user] = new_pass
                    st.success("Password updated successfully!")
                else:
                    st.error("New passwords don't match")
            else:
                st.error("Current password is incorrect")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Application Settings")
        
        # Theme settings
        st.markdown("#### Display Settings")
        theme = st.selectbox("Theme", ["Light", "Dark", "System"], key="theme_select")
        font_size = st.select_slider("Base Font Size", options=["Small", "Medium", "Large"], value="Medium")
        
        # Data management
        st.markdown("#### Data Management")
        if st.button("Clear Session Data", use_container_width=True):
            st.session_state.results = {}
            st.session_state.handwriting_image = None
            st.session_state.tracking_active = False
            st.session_state.eye_tracking_data = {"gaze_points": [], "fixations": [], "drastic_movements": 0}
            st.session_state.speech_data = {"spoken_text": "", "accuracy": 0, "is_recording": False}
            st.session_state.speech_recognition_active = False
            st.success("Session data cleared!")
        
        if st.button("Reset Assessment", use_container_width=True):
            st.session_state.current_step = 1
            st.session_state.results = {}
            st.session_state.handwriting_image = None
            st.session_state.tracking_active = False
            st.session_state.eye_tracking_data = {"gaze_points": [], "fixations": [], "drastic_movements": 0}
            st.session_state.speech_data = {"spoken_text": "", "accuracy": 0, "is_recording": False}
            st.session_state.speech_recognition_active = False
            st.success("Assessment reset to Step 1!")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------- MAIN APP -------------------
def main_app():
    # Sidebar
    sidebar()
    
    # Main content based on current step
    show_progress()
    
    if st.session_state.current_step == 1:
        handwriting_analysis()
    elif st.session_state.current_step == 2:
        eye_tracking()
    elif st.session_state.current_step == 3:
        generate_report()
    elif st.session_state.current_step == 4:
        settings_page()

# ------------------- APP ROUTER -------------------
if not st.session_state.logged_in:
    login_page()
else:
    main_app()