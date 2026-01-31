# 🧠 Multimodal Learning Difficulty Assessment App

A **Streamlit-based intelligent assessment system** that performs **simultaneous eye tracking and speech analysis** to identify potential indicators of **Dyslexia and Dysgraphia** during a reading task.

This application integrates **computer vision**, **speech processing**, and **machine learning** to provide a non-invasive, real-time screening tool for learning difficulties.

---

## 🚀 Features

### 👁️ Eye Tracking Analysis
- Real-time webcam-based eye tracking using **MediaPipe Face Mesh**
- Gaze point detection using iris landmarks
- Fixation detection and duration analysis
- Detection of **eyes-off-screen events**
- Eye movement feature extraction for dyslexia prediction

### 🎤 Speech Analysis
- Live audio recording while reading a passage
- Speech-to-text conversion
- Speech feature extraction (rate, pauses, pitch variation, voice breaks)
- Dysgraphia prediction based on speech patterns
- Display of recognized **spoken text**

### 🧠 Combined Multimodal Assessment
- Simultaneous eye tracking + speech recording
- Unified Start/Stop control
- Independent analysis of both modalities
- Combined risk interpretation
- User-friendly visual summaries

---

## 🖥️ Application Workflow

1. Select a reading passage
2. Click **Start Dual Analysis**
3. Read the passage aloud while looking at the screen
4. Click **Stop Dual Analysis**
5. View:
   - Eye tracking metrics
   - Speech analysis results
   - Dyslexia & Dysgraphia predictions
   - Combined assessment summary

---

## 📊 Output Metrics

### Eye Tracking
- Number of fixations
- Mean & standard deviation of fixation duration
- Gaze dispersion (X & Y)
- Number of times eyes moved off screen
- Dyslexia prediction with confidence

### Speech Analysis
- Recognized spoken text
- Speech rate (WPM)
- Pause ratio
- Pitch variation
- Voice break ratio
- Dysgraphia prediction with confidence

---

## 🛠️ Technologies Used

### Frontend
- **Streamlit** – Web application framework

### Computer Vision
- **OpenCV**
- **MediaPipe**

### Speech & Audio Processing
- **Librosa**
- **Sounddevice / PyAudio**
- **Speech Recognition APIs**

### Machine Learning & Data Processing
- **NumPy**
- **SciPy**
- **Scikit-learn**

### Visualization
- **Matplotlib**

---

## 📦 Required Libraries

Install all dependencies using:

```bash
pip install streamlit opencv-python mediapipe numpy scipy scikit-learn librosa matplotlib sounddevice pyaudio
⚠️ Note (Windows users)
If pyaudio installation fails:

pip install pipwin
pipwin install pyaudio
```
## How to Run the App
```bash
streamlit run fin.py
```
## Ensure:
- Webcam is connected
- Microphone is enabled
- App is run in a quiet environment for better speech accuracy