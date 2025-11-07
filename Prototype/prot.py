import cv2
import whisper
import sounddevice as sd
import soundfile as sf
import mediapipe as mp
import threading
import time
import numpy as np
from deepface import DeepFace
import google.generativeai as genai
import os
from dotenv import load_dotenv

AUDIO_FILENAME = "speech.wav"
DURATION = 20  
SAMPLERATE = 16000  


load_dotenv()


AUDIO_FILE = "speech.wav"
VIDEO_FILE = "speech.avi"
FACE_SNAPSHOT = "face_snapshot.jpg"
DURATION = 20  # seconds
SAMPLERATE = 16000
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20.0

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def record_audio():
    print("\nAudio recording started...")
    audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1)
    sd.wait()
    sf.write(AUDIO_FILE, audio, SAMPLERATE)
    print("Audio recording complete.")

def is_looking_away(landmarks, image_width):
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    left_eye_x = landmarks[LEFT_EYE_OUTER].x * image_width
    right_eye_x = landmarks[RIGHT_EYE_OUTER].x * image_width
    eye_distance = abs(right_eye_x - left_eye_x)
    eye_center = (left_eye_x + right_eye_x) / 2
    frame_center = image_width / 2
    deviation = abs(eye_center - frame_center) / eye_distance
    return deviation > 0.3

# === Video Recorder + Analysis ===
def record_video(feedback_dict):
    print("Video recording started...")
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(VIDEO_FILE, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

    start_time = time.time()
    total_frames = 0
    looking_away_frames = 0
    hands_near_face_frames = 0

    while time.time() - start_time < DURATION:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        image_height, image_width, _ = frame.shape

        face_results = face_mesh.process(rgb)
        hand_results = hands_detector.process(rgb)
        rgb.flags.writeable = True

        # Face mesh + snapshot
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                )
                if is_looking_away(face_landmarks.landmark, image_width):
                    looking_away_frames += 1
                if total_frames == 10:
                    cv2.imwrite(FACE_SNAPSHOT, frame)

        # Hand detection
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )
            hands_near_face_frames += 1

        out.write(frame)
        total_frames += 1

        # Live preview
        cv2.imshow("Live Camera (Press Q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video recording complete.")

    feedback_dict['face_away_ratio'] = looking_away_frames / total_frames
    feedback_dict['hands_near_face_ratio'] = hands_near_face_frames / total_frames

    try:
        emotion_result = DeepFace.analyze(img_path=FACE_SNAPSHOT, actions=["emotion"], enforce_detection=False)
        feedback_dict['emotion'] = emotion_result[0]['dominant_emotion']
    except Exception as e:
        feedback_dict['emotion'] = "Not detected"
        print(f"[Emotion Detection Error] {e}")

# === Transcription ===
def transcribe():
    print("Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(AUDIO_FILE)
    return result["text"]

# === Grammar Feedback ===
def correct_grammar_with_gemini(transcript):
    prompt = f"""
You are an English language expert. Given the following transcript from a speech, return:
1. A grammatically corrected version.
2. A one-line constructive feedback if needed.

Transcript:
\"\"\"{transcript}\"\"\"
"""
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

# === Feedback Report ===
def feedback_report(transcript, feedback_dict):
    print("\n--- FEEDBACK REPORT ---")
    print(f"Transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
    print(f"Face looking away ratio: {feedback_dict['face_away_ratio']:.2f}")
    print(f"Hands near face ratio: {feedback_dict['hands_near_face_ratio']:.2f}")
    print(f"Emotion detected: {feedback_dict.get('emotion', 'N/A')}")

    if feedback_dict['face_away_ratio'] > 0.3:
        print("You looked away often — try to maintain more eye contact.")
    else:
        print("Great job maintaining eye contact.")

    if feedback_dict['hands_near_face_ratio'] > 0.2:
        print("You might be fidgeting — try keeping your hands calm.")
    else:
        print("Calm and confident hand posture.")

    print("\n--- Grammar Feedback ---")
    grammar_feedback = correct_grammar_with_gemini(transcript)
    print(grammar_feedback)

# === Main Execution ===
if __name__ == "__main__":
    feedback = {}

    audio_thread = threading.Thread(target=record_audio)
    video_thread = threading.Thread(target=record_video, args=(feedback,))

    audio_thread.start()
    video_thread.start()

    audio_thread.join()
    video_thread.join()

    transcript = transcribe()
    feedback_report(transcript, feedback)
