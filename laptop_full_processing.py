import io
import time
import logging
from threading import Condition, Thread, Lock
import traceback # Import for detailed error tracebacks

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import google.generativeai as genai
# from gtts import gTTS # REMOVED TTS
import os

from flask import Flask, Response, render_template_string, jsonify, request
from flask_cors import CORS

# --- Configuration ---
PI_IP_ADDRESS = "192.168.220.16"  # <--- *** REPLACE WITH YOUR PI's ACTUAL IP ADDRESS ***
PI_STREAM_PORT = 3000
PI_STREAM_URL = f"http://{PI_IP_ADDRESS}:{PI_STREAM_PORT}/video_feed"

LAPTOP_FLASK_PORT = 5001 # Port this laptop server runs on

# --- Copied Config/Inits (Model, Gemini, Signs) ---
GOOGLE_API_KEY = "" # Replace if needed
if not GOOGLE_API_KEY: print("Laptop Warning: Google API Key not found.")

MODEL_PATH = "model.tflite"
TRAIN_CSV_PATH = "train.csv"
EXAMPLE_PARQUET_PATH = "10042041.parquet" # Needed only for skeleton structure loading
ROWS_PER_FRAME = 543
PREDICTION_INTERVAL_SECONDS = 3 # Adjust based on laptop speed
# SPEECH_TEMP_FILE = "/tmp/speech.mp3" # REMOVED TTS
CAMERA_RESOLUTION = (640, 480) # Define here for HTML template

# Initialize Gemini
gemini_model = None
if GOOGLE_API_KEY:
    try: genai.configure(api_key=GOOGLE_API_KEY); gemini_model = genai.GenerativeModel('gemini-pro'); print("Laptop: Gemini initialized.")
    except Exception as e: print(f"Laptop: Error initializing Gemini: {e}")
else: print("Laptop: Google API Key not set.")

# Initialize TFLite
interpreter = None
pred_fn = None
try: interpreter = tf.lite.Interpreter(model_path=MODEL_PATH); interpreter.allocate_tensors(); pred_fn = interpreter.get_signature_runner("serving_default"); print(f"Laptop: TFLite model '{MODEL_PATH}' loaded.")
except Exception as e: print(f"Laptop FATAL: Error loading TFLite model: {e}"); exit()

# Load sign mappings
SIGN2ORD = None
ORD2SIGN = None
try: train = pd.read_csv(TRAIN_CSV_PATH); train['sign_ord'] = train['sign'].astype('category').cat.codes; SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict(); ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict(); print(f"Laptop: Sign mappings loaded from '{TRAIN_CSV_PATH}'.")
except Exception as e: print(f"Laptop FATAL: Error loading train data: {e}"); exit()

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = None # Will init in thread

# Load Skeleton Structure (needed for landmark extraction order)
xyz_skeleton = None
try: xyz = pd.read_parquet(EXAMPLE_PARQUET_PATH, columns=['type', 'landmark_index']); xyz_skeleton = xyz.iloc[:ROWS_PER_FRAME][["type", "landmark_index"]].drop_duplicates().reset_index(drop=True); xyz_skeleton['row_id'] = xyz_skeleton.index; print(f"Laptop: Loaded skeleton structure with {len(xyz_skeleton)} definitions.")
except Exception as e: print(f"Laptop FATAL: Error loading skeleton: {e}"); exit()

# Landmark extraction function (same as before)
def extract_landmarks_as_numpy(results, xyz_skel):
    landmarks = []
    def add_landmarks(landmark_list, landmark_type):
        if landmark_list:
            for i, point in enumerate(landmark_list.landmark): landmarks.append({"type": landmark_type, "landmark_index": i, "x": point.x, "y": point.y, "z": point.z})
    add_landmarks(results.face_landmarks, "face"); add_landmarks(results.pose_landmarks, "pose"); add_landmarks(results.left_hand_landmarks, "left_hand"); add_landmarks(results.right_hand_landmarks, "right_hand")
    if not landmarks: return np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)
    landmarks_df = pd.DataFrame(landmarks); merged_df = xyz_skel.merge(landmarks_df, on=["type", "landmark_index"], how="left"); merged_df = merged_df.sort_values(by='row_id'); landmark_values = merged_df[["x", "y", "z"]].fillna(np.nan).values.astype(np.float32)
    if len(landmark_values) < ROWS_PER_FRAME: padding = np.full((ROWS_PER_FRAME - len(landmark_values), 3), np.nan, dtype=np.float32); landmark_values = np.vstack((landmark_values, padding))
    return landmark_values[:ROWS_PER_FRAME, :]


# --- TTS Function (REMOVED) ---
# def speak_word(text):
#     ...

# --- Gemini API Call Function (same as before) ---
def get_display_message_from_api(recognised_words):
    global display_message # Allow updating global status
    if not gemini_model: msg = f"Recognized: {' '.join(recognised_words)}"; display_message = msg; return msg
    if not recognised_words: msg = "No words recognized"; display_message = msg; return msg
    filtered_words = [w for w in recognised_words if w.lower() != "tv"];
    if not filtered_words: msg = "Only 'TV' recognized"; display_message = msg; return msg
    prompt = f"Objective:\nConstruct a simple, coherent English sentence from these ASL words: {filtered_words}\nOutput Sentence:"
    display_message = "Generating sentence..."
    try: response = gemini_model.generate_content(prompt); generated_text = response.text.strip(); display_message = f"Sentence: {generated_text}"; return generated_text
    except Exception as e: print(f"Laptop Gemini API Error: {e}"); msg = f"API Error. Recognized: {' '.join(filtered_words)}"; display_message = msg; return msg


# --- Streaming Output Class (Laptop - for processed frames) ---
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None # Holds the *processed* JPEG bytes
        self.condition = Condition()
    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

# --- Global State (Laptop) ---
output = StreamingOutput() # For the *processed* MJPEG stream served by this laptop
stop_processing_flag = False
processing_thread = None
holistic = None # MediaPipe instance

# Prediction state
collected_frame_data = []
last_prediction_time = 0
current_sign_prediction = "Connecting to Pi..."
unique_signs_in_batch = []
display_message = "Initializing..."
# last_spoken_word = None # REMOVED TTS
data_lock = Lock() # Lock for thread-safe access to shared prediction state

# --- Video Capture, Processing, Prediction Thread ---
def run_capture_process_predict():
    global holistic, stop_processing_flag, output
    global collected_frame_data, last_prediction_time, current_sign_prediction
    global unique_signs_in_batch, display_message #, last_spoken_word # REMOVED TTS

    cap = None
    connect_retry_delay = 5
    frame_counter = 0 # DEBUG

    try:
        print("Laptop Thread: Initializing MediaPipe Holistic...")
        holistic = mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2)
        print("Laptop Thread: MediaPipe Initialized.")
        last_prediction_time = time.time() # Initialize timer

        while not stop_processing_flag:
            # --- Connect/Reconnect to Pi Stream ---
            if cap is None or not cap.isOpened():
                print(f"Laptop Thread: Attempting to connect to Pi stream: {PI_STREAM_URL}")
                if cap: cap.release()
                cap = cv2.VideoCapture(PI_STREAM_URL)
                if not cap.isOpened():
                    print(f"Laptop Thread: Failed to connect to Pi stream. Retrying in {connect_retry_delay}s...")
                    cap = None
                    with data_lock: display_message = f"Cannot connect to Pi stream ({PI_IP_ADDRESS}). Retrying..."; current_sign_prediction = "Disconnected"
                    time.sleep(connect_retry_delay)
                    continue
                else:
                     print("Laptop Thread: Connected to Pi stream.")
                     with data_lock: display_message = "Pi connected. Processing..."

            # --- Read Frame from Pi Stream ---
            # print("Laptop DBG: Reading frame...") # DEBUG: Can be noisy
            ret, frame_bgr = cap.read()
            if not ret:
                print("Laptop Thread: Failed to grab frame from Pi stream. Reconnecting...")
                cap.release(); cap = None
                with data_lock: display_message = "Lost connection to Pi stream. Reconnecting..."; current_sign_prediction = "Disconnected"
                time.sleep(1)
                continue
            frame_counter += 1
            print(f"Laptop DBG: Received frame {frame_counter} - Shape: {frame_bgr.shape if frame_bgr is not None else 'None'}") # DEBUG

            # --- Process Frame ---
            try:
                if frame_bgr is None: # Add check for None frame
                    print("Laptop DBG: Frame is None, skipping processing.")
                    continue

                # 1. Convert BGR to RGB for MediaPipe
                print("Laptop DBG: Converting to RGB") # DEBUG
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False # Optimization

                # 2. MediaPipe Processing
                print("Laptop DBG: Processing MediaPipe...") # DEBUG
                start_mp_time = time.time()
                results = holistic.process(frame_rgb)
                end_mp_time = time.time()
                print(f"Laptop DBG: MediaPipe Processed (took {end_mp_time - start_mp_time:.4f}s)") # DEBUG
                frame_rgb.flags.writeable = True

                # 3. Landmark Extraction for Prediction
                print("Laptop DBG: Extracting Landmarks") # DEBUG
                landmarks_np = extract_landmarks_as_numpy(results, xyz_skeleton)
                landmarks_found = not np.all(np.isnan(landmarks_np))
                print(f"Laptop DBG: Landmarks Extracted, Shape: {landmarks_np.shape}, Found: {landmarks_found}") # DEBUG

                with data_lock:
                     if landmarks_found:
                          collected_frame_data.append(landmarks_np)

                # 4. Draw Landmarks for Laptop's Streaming Feed
                print("Laptop DBG: Drawing Landmarks...") # DEBUG
                frame_to_stream = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # Convert back for drawing/encoding
                try:
                    print("Laptop DBG: Drawing Face") # DEBUG
                    mp_drawing.draw_landmarks(frame_to_stream, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, None, mp_drawing_styles.get_default_face_mesh_contours_style())
                    print("Laptop DBG: Drawing Pose") # DEBUG
                    mp_drawing.draw_landmarks(frame_to_stream, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, None, mp_drawing_styles.get_default_pose_landmarks_style())
                    print("Laptop DBG: Drawing Left Hand") # DEBUG
                    mp_drawing.draw_landmarks(frame_to_stream, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    print("Laptop DBG: Drawing Right Hand") # DEBUG
                    mp_drawing.draw_landmarks(frame_to_stream, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    print("Laptop DBG: Landmarks Drawn OK") # DEBUG
                except Exception as draw_e:
                    print(f"\n----- ERROR DURING LANDMARK DRAWING (Frame {frame_counter}) -----")
                    print(f"DRAW ERROR TYPE: {type(draw_e)}")
                    print(f"DRAW ERROR DETAILS: {draw_e}")
                    # If the error object IS the tuple, print it directly for confirmation
                    if isinstance(draw_e, tuple):
                        print(f"Attempting to draw connection: {draw_e}")
                    # Print full traceback for drawing errors
                    traceback.print_exc()
                    print("-----------------------------------------\n")
                    # Continue processing frame, drawing might be incomplete

                # 5. Encode *Processed* Frame for Laptop's Stream
                print("Laptop DBG: Encoding Frame") # DEBUG
                is_success, buffer = cv2.imencode(".jpg", frame_to_stream)
                if is_success:
                    output.write(buffer.tobytes())
                    print("Laptop DBG: Frame Encoded and Written") # DEBUG
                else:
                    print(f"Laptop Thread: Error encoding frame {frame_counter} to JPEG")

                # 6. TFLite Prediction (Periodically) - Use lock around shared state
                current_time = time.time()
                if current_time - last_prediction_time >= PREDICTION_INTERVAL_SECONDS:
                    print(f"Laptop DBG: Checking prediction (Frame {frame_counter})") # DEBUG
                    # Make copies of shared data under lock
                    local_collected_data = []
                    with data_lock:
                        local_collected_data = list(collected_frame_data)
                        collected_frame_data.clear()

                    if local_collected_data:
                        print(f"Laptop DBG: Predicting with {len(local_collected_data)} frames.") # DEBUG
                        input_data = np.stack(local_collected_data, axis=0)
                        input_data = np.nan_to_num(input_data.astype(np.float32), nan=0.0)
                        try:
                            prediction = pred_fn(inputs=input_data)
                            sign_ord = prediction['outputs'].argmax()
                            sign_name = ORD2SIGN.get(sign_ord, "Unknown Sign")
                            print(f"Laptop DBG: Prediction result: {sign_name} (ord: {sign_ord})") # DEBUG

                            # Update state under lock
                            with data_lock:
                                current_sign_prediction = sign_name
                                # TTS Logic REMOVED
                                # Collect for Sentence
                                if sign_name != "TV" and sign_name != "Unknown Sign":
                                    if sign_name not in unique_signs_in_batch:
                                        unique_signs_in_batch.append(sign_name)
                                # Update display message
                                display_message = f"Prediction: {current_sign_prediction} | Collected: {', '.join(unique_signs_in_batch)}"

                        except Exception as pred_e:
                            print(f"Laptop Thread: Error during prediction: {pred_e}")
                            traceback.print_exc() # Print prediction error traceback
                            with data_lock:
                                current_sign_prediction = "Prediction Error"
                                display_message = "Error during prediction."
                                # last_spoken_word = None # REMOVED TTS
                    else:
                         print("Laptop DBG: No landmark data collected for prediction.") # DEBUG
                         with data_lock:
                              current_sign_prediction = "No Movement Detected"
                              display_message = f"Prediction: {current_sign_prediction} | Collected: {', '.join(unique_signs_in_batch)}"
                              # last_spoken_word = None # REMOVED TTS

                    # Update prediction timer (outside lock)
                    last_prediction_time = current_time

            except Exception as e:
                # This is the CATCH-ALL for other errors in the loop (outside drawing block)
                print(f"\n----- ERROR in main processing loop (Frame {frame_counter}, NOT DRAWING/PREDICTION) -----")
                print(f"ERROR TYPE: {type(e)}")
                print(f"ERROR DETAILS: {e}")
                traceback.print_exc() # Print the full traceback here
                print("-------------------------------------------------------\n")
                # Decide if you want to stop the thread on such errors
                # stop_processing_flag = True # Option: Stop thread on any error

    except Exception as e:
        # Errors during thread init or fatal connection issues
        print(f"Laptop Thread: FATAL ERROR in setup or main loop: {e}")
        traceback.print_exc() # Print fatal error traceback
        with data_lock: display_message = f"FATAL ERROR: {e}"
        stop_processing_flag = True # Signal main thread to stop
    finally:
        print("Laptop Thread: Cleaning up...")
        if cap: cap.release()
        if holistic: holistic.close()
        # TTS Cleanup REMOVED
        # if os.path.exists(SPEECH_TEMP_FILE):
        #    try: os.remove(SPEECH_TEMP_FILE)
        #    except Exception as e: print(f"Laptop Thread: Error removing temp speech file: {e}")
        print("Laptop Thread: Exiting.")


# --- Flask App (Laptop - Serves UI, Processed Stream, Status, Control) ---
laptop_app = Flask(__name__)
CORS(laptop_app) # Enable CORS

@laptop_app.route('/')
def index():
    """ Serves the main HTML page. """
    # JavaScript now fetches from this server's own endpoints
    return render_template_string("""
    <html>
    <head>
        <title>Laptop ASL Recognition</title>
        <script>
            function updateStatus() {
                fetch('/status') // Fetch from this laptop server
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('prediction').innerText = data.prediction;
                        document.getElementById('message').innerText = data.message;
                        document.getElementById('collected').innerText = data.collected.join(', ');
                    })
                    .catch(error => {
                        console.error('Error fetching status:', error);
                        document.getElementById('message').innerText = 'Error fetching status.';
                    });
            }
            function generateSentence() {
                fetch('/generate_sentence', { method: 'POST' }) // POST to this laptop server
                    .then(response => response.json())
                    .then(data => {
                        console.log("Generate sentence response:", data);
                        setTimeout(updateStatus, 500);
                    })
                    .catch(error => console.error('Error generating sentence:', error));
                 setTimeout(updateStatus, 100);
            }
            setInterval(updateStatus, 2000);
        </script>
    </head>
    <body onload="updateStatus()">
        <h1>Laptop ASL Recognition (Processing Pi Stream)</h1>
        <!-- Video stream comes from THIS laptop server's /video_feed -->
        <img src="{{ url_for('video_feed') }}" width="{{w}}" height="{{h}}">
        <h2>Status</h2>
        <p><strong>Prediction:</strong> <span id="prediction">Loading...</span></p>
        <p><strong>Message:</strong> <span id="message">Initializing...</span></p>
        <p><strong>Collected Words:</strong> <span id="collected"></span></p>
        <button onclick="generateSentence()">Generate Sentence</button>
    </body>
    </html>
    """, w=CAMERA_RESOLUTION[0], h=CAMERA_RESOLUTION[1]) # Pass resolution for img tag

def gen_processed_frames():
    """ Generator for the PROCESSED MJPEG stream. """
    global output
    while True:
        if stop_processing_flag: break
        if not output: time.sleep(0.1); continue
        with output.condition:
            signalled = output.condition.wait(timeout=1.0)
            if not signalled:
                # print("Laptop DBG: Timeout waiting for processed frame") # DEBUG: Can be noisy
                continue
            frame = output.frame # Get the latest processed frame
        if frame is None: continue
        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except GeneratorExit: break
        except Exception as e: print(f"Laptop: Error yielding processed frame: {e}"); break
    print("Laptop: Processed frame generation stopped.")

@laptop_app.route('/video_feed')
def video_feed():
    """ Processed video streaming route. """
    return Response(gen_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@laptop_app.route('/status')
def status():
    """ Provides current prediction and status message. """
    with data_lock: # Access shared state safely
        # Create a copy of the list for thread safety when serializing
        collected_copy = list(unique_signs_in_batch)
        return jsonify({
            'prediction': current_sign_prediction,
            'message': display_message,
            'collected': collected_copy
        })

@laptop_app.route('/generate_sentence', methods=['POST'])
def generate_sentence_api():
    """ Triggers the Gemini API call. """
    global unique_signs_in_batch, display_message, data_lock
    print("Laptop API: /generate_sentence called")
    words_to_process = []
    with data_lock:
        words_to_process = list(unique_signs_in_batch)
        unique_signs_in_batch.clear()
        display_message = "Requesting sentence generation..."
    sentence = get_display_message_from_api(words_to_process) # Call Gemini outside lock
    print("Laptop API: Generated sentence:", sentence)
    return jsonify({'status': 'Processing sentence request', 'result': sentence})

# --- Main Execution (Laptop) ---
if __name__ == '__main__':
    processing_thread = None
    flask_thread = None
    try:
        print("Laptop Main: Starting processing thread...")
        processing_thread = Thread(target=run_capture_process_predict, daemon=True)
        processing_thread.start()

        print("Laptop Main: Waiting for processing thread initialization...")
        time.sleep(4) # Allow time for MediaPipe init and first connection attempt

        if not processing_thread.is_alive():
            print("Laptop Main: Processing thread failed to start. Exiting.")
        else:
            print(f"Laptop Main: Starting Flask server on port {LAPTOP_FLASK_PORT}...")
            flask_thread = Thread(target=lambda: laptop_app.run(host='0.0.0.0', port=LAPTOP_FLASK_PORT, debug=False), daemon=True)
            flask_thread.start()

            while True:
                 if not processing_thread.is_alive() or not flask_thread.is_alive():
                      print("Laptop Main: A required thread has stopped. Exiting.")
                      stop_processing_flag = True
                      break
                 time.sleep(1)

    except KeyboardInterrupt:
        print("\nLaptop Main: Ctrl+C pressed.")
    finally:
        print("Laptop Main: Setting stop flag...")
        stop_processing_flag = True

        if processing_thread and processing_thread.is_alive():
            print("Laptop Main: Waiting for processing thread...")
            processing_thread.join(timeout=5)
            if processing_thread.is_alive(): print("Laptop Main: Warning: Processing thread didn't exit cleanly.")

        print("Laptop Main: Cleanup complete.")