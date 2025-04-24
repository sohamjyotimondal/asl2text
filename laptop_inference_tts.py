import socket
import threading
import time
import json
import struct
import numpy as np
import pandas as pd # For loading sign mappings
import tensorflow as tf
import google.generativeai as genai
from gtts import gTTS
import os

from flask import Flask, jsonify, request # Added request for POST
from flask_cors import CORS


# --- Configuration ---
LISTENING_IP = "0.0.0.0"  # Listen on all available network interfaces
LISTENING_PORT = 9999     # Port the Pi connects to for landmarks
FLASK_PORT = 5001         # Port for the status API server (must match Pi's JS)

# --- Copied Config/Inits from original App (Needed for Inference/API/TTS) ---
GOOGLE_API_KEY = "AIzaSyD2yKNUkzYqNobXA-ACKvEGkyap1dSPOYs" # Replace if needed
if not GOOGLE_API_KEY: print("Laptop Warning: Google API Key not found.")

MODEL_PATH = "model.tflite" # Needs model.tflite on laptop
TRAIN_CSV_PATH = "train.csv" # Needs train.csv on laptop
ROWS_PER_FRAME = 543
PREDICTION_INTERVAL_SECONDS = 3
SPEECH_TEMP_FILE = "/tmp/speech.mp3" # Or use a path suitable for the laptop OS

# Initialize Gemini
gemini_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
        print("Laptop: Google Gemini initialized.")
    except Exception as e: print(f"Laptop: Error initializing Gemini: {e}")
else: print("Laptop: Google API Key not set.")

# Initialize TFLite
interpreter = None
pred_fn = None
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    pred_fn = interpreter.get_signature_runner("serving_default")
    print(f"Laptop: TFLite model '{MODEL_PATH}' loaded.")
except Exception as e: print(f"Laptop FATAL: Error loading TFLite model: {e}"); exit()

# Load sign mappings
SIGN2ORD = None
ORD2SIGN = None
try:
    train = pd.read_csv(TRAIN_CSV_PATH)
    train['sign_ord'] = train['sign'].astype('category').cat.codes
    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
    print(f"Laptop: Sign mappings loaded from '{TRAIN_CSV_PATH}'.")
except Exception as e: print(f"Laptop FATAL: Error loading train data: {e}"); exit()


# --- Global State (Laptop) ---
stop_server_flag = False
connected_pi_addr = None

# Prediction state
collected_frame_data = []
last_prediction_time = 0
current_sign_prediction = "Waiting for Pi..."
unique_signs_in_batch = []
display_message = "Waiting for Pi connection..."
last_spoken_word = None

# --- TTS Function (same as before, runs on Laptop) ---
def speak_word(text):
    if not text: return
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(SPEECH_TEMP_FILE)
        print(f"Laptop TTS: Playing '{text}'...")
        # Use appropriate player for laptop OS
        # Linux: mpg123 -q, macOS: afplay, Windows: ??? (might need playsound library)
        if os.name == 'posix': # Linux or macOS
            player_cmd = "mpg123 -q" if os.uname().sysname == "Linux" else "afplay"
            os.system(f"{player_cmd} {SPEECH_TEMP_FILE}")
        else: # Basic fallback for Windows (install playsound: pip install playsound==1.2.2)
            try:
                 from playsound import playsound
                 playsound(SPEECH_TEMP_FILE)
            except ImportError:
                 print("Laptop TTS Warning: 'playsound' module not found for Windows. Install it: pip install playsound==1.2.2")
            except Exception as ps_e:
                 print(f"Laptop TTS Error (playsound): {ps_e}")

        # if os.path.exists(SPEECH_TEMP_FILE): os.remove(SPEECH_TEMP_FILE)
    except Exception as e:
        print(f"Laptop TTS Error: Failed to speak '{text}': {e}")


# --- Gemini API Call Function (same as before, runs on Laptop) ---
def get_display_message_from_api(recognised_words):
    global display_message
    if not gemini_model:
        msg = f"Recognized: {' '.join(recognised_words)}"
        display_message = msg; return msg
    if not recognised_words:
        msg = "No words recognized"; display_message = msg; return msg
    filtered_words = [w for w in recognised_words if w.lower() != "tv"]
    if not filtered_words:
        msg = "Only 'TV' recognized"; display_message = msg; return msg
    prompt = f"Objective:\nConstruct a simple, coherent English sentence from these ASL words: {filtered_words}\nOutput Sentence:"
    display_message = "Generating sentence..."
    try:
        response = gemini_model.generate_content(prompt)
        generated_text = response.text.strip()
        display_message = f"Sentence: {generated_text}"; return generated_text
    except Exception as e:
        print(f"Laptop Gemini API Error: {e}")
        msg = f"API Error. Recognized: {' '.join(filtered_words)}"
        display_message = msg; return msg

# --- Landmark Receiving and Processing Thread ---
def handle_pi_connection(conn, addr):
    global connected_pi_addr, collected_frame_data, last_prediction_time
    global current_sign_prediction, unique_signs_in_batch, display_message, last_spoken_word

    connected_pi_addr = addr
    print(f"Laptop: Connection accepted from {addr}")
    display_message = "Pi connected. Receiving landmarks..."
    last_prediction_time = time.time() # Start timer on connect
    conn.settimeout(10.0) # Timeout for receiving data

    try:
        while not stop_server_flag:
            # 1. Receive message length (4 bytes)
            len_bytes = conn.recv(4)
            if not len_bytes:
                print("Laptop: Pi disconnected (no length received).")
                break
            message_len = struct.unpack('>I', len_bytes)[0]

            # 2. Receive the actual message data
            message_bytes = b''
            while len(message_bytes) < message_len:
                chunk = conn.recv(min(message_len - len(message_bytes), 4096))
                if not chunk:
                    raise ConnectionError("Laptop: Pi disconnected during message receive.")
                message_bytes += chunk

            # 3. Deserialize landmarks
            json_string = message_bytes.decode('utf-8')
            received_list = json.loads(json_string)
            landmarks_np = np.array(received_list, dtype=np.float32)

            # Check if received data seems valid (correct shape)
            if landmarks_np.shape == (ROWS_PER_FRAME, 3):
                 if not np.all(np.isnan(landmarks_np)):
                     collected_frame_data.append(landmarks_np)
                 # print(f"Laptop: Received frame {len(collected_frame_data)}") # Debug: noisy
            else:
                 print(f"Laptop Warning: Received landmark data with unexpected shape {landmarks_np.shape}")


            # 4. Run Prediction Periodically
            current_time = time.time()
            if current_time - last_prediction_time >= PREDICTION_INTERVAL_SECONDS:
                if collected_frame_data:
                    input_data = np.stack(collected_frame_data, axis=0)
                    input_data = np.nan_to_num(input_data.astype(np.float32), nan=0.0)
                    try:
                        prediction = pred_fn(inputs=input_data)
                        sign_ord = prediction['outputs'].argmax()
                        sign_name = ORD2SIGN.get(sign_ord, "Unknown Sign")
                        current_sign_prediction = sign_name

                        # TTS Logic
                        valid_sign = (sign_name != "Unknown Sign" and sign_name != "No Movement Detected" and sign_name.lower() != "tv")
                        if valid_sign and sign_name != last_spoken_word:
                            speak_word(sign_name) # Consider threading this?
                            last_spoken_word = sign_name

                        # Collect for Sentence
                        if sign_name != "TV" and sign_name != "Unknown Sign":
                            if sign_name not in unique_signs_in_batch:
                                unique_signs_in_batch.append(sign_name)
                        display_message = f"Prediction: {current_sign_prediction} | Collected: {', '.join(unique_signs_in_batch)}"

                    except Exception as e:
                        print(f"Laptop: Error during prediction: {e}")
                        current_sign_prediction = "Prediction Error"
                        display_message = "Error during prediction."
                        last_spoken_word = None
                    finally:
                       collected_frame_data = [] # Clear batch regardless of success/failure
                       last_prediction_time = current_time
                else:
                    # No landmarks collected, update status
                    current_sign_prediction = "No Movement Detected"
                    display_message = f"Prediction: {current_sign_prediction} | Collected: {', '.join(unique_signs_in_batch)}"
                    last_prediction_time = current_time


    except socket.timeout:
        print(f"Laptop: Socket timeout waiting for data from {addr}.")
    except (ConnectionError, ConnectionResetError, struct.error, json.JSONDecodeError) as e:
        print(f"Laptop: Connection error with {addr}: {e}")
    except Exception as e:
        print(f"Laptop: Unhandled error in connection handler: {e}")
    finally:
        print(f"Laptop: Closing connection from {addr}")
        conn.close()
        connected_pi_addr = None
        # Reset status when Pi disconnects
        current_sign_prediction = "Waiting for Pi..."
        display_message = "Pi disconnected. Waiting for connection..."
        collected_frame_data = []
        unique_signs_in_batch = []
        last_spoken_word = None


# --- Socket Server Thread ---
def run_socket_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow reusing address quickly
    try:
        server_socket.bind((LISTENING_IP, LISTENING_PORT))
        server_socket.listen(1) # Listen for one connection (the Pi)
        print(f"Laptop: Landmark socket server listening on {LISTENING_IP}:{LISTENING_PORT}")

        while not stop_server_flag:
            server_socket.settimeout(1.0) # Check stop flag periodically
            try:
                conn, addr = server_socket.accept()
                # Start a new thread to handle this specific connection
                # This allows the server to listen for new connections if the current one drops
                client_handler_thread = threading.Thread(target=handle_pi_connection, args=(conn, addr), daemon=True)
                client_handler_thread.start()
            except socket.timeout:
                continue # No connection attempt, just loop again
            except Exception as e:
                print(f"Laptop: Error accepting connection: {e}")


    except Exception as e:
        print(f"Laptop: Socket server error: {e}")
    finally:
        print("Laptop: Closing socket server.")
        server_socket.close()
        # Clean up temp speech file if needed
        if os.path.exists(SPEECH_TEMP_FILE):
            try: os.remove(SPEECH_TEMP_FILE)
            except Exception as e: print(f"Laptop: Error removing temp speech file: {e}")


# --- Flask App (Laptop - for Status and Control) ---
laptop_app = Flask(__name__)
CORS(laptop_app)

@laptop_app.route('/status')
def status():
    # Access globals (use locks if modifying state here, but reads are okay for simple types)
    return jsonify({
        'prediction': current_sign_prediction,
        'message': display_message,
        'collected': unique_signs_in_batch
    })

@laptop_app.route('/generate_sentence', methods=['POST'])
def generate_sentence_api():
    global unique_signs_in_batch, display_message
    print("Laptop API: /generate_sentence called")
    words_to_process = list(unique_signs_in_batch)
    unique_signs_in_batch = [] # Clear collected words
    display_message = "Requesting sentence generation..." # Update message shown in UI
    sentence = get_display_message_from_api(words_to_process) # Call the function
    print("Laptop API: Generated sentence:", sentence)
    return jsonify({'status': 'Processing sentence request', 'result': sentence})


# --- Main Execution (Laptop) ---
if __name__ == '__main__':
    socket_thread = None
    flask_thread = None
    try:
        # Start socket server in a thread
        print("Laptop Main: Starting socket server thread...")
        socket_thread = threading.Thread(target=run_socket_server, daemon=True)
        socket_thread.start()

        # Start Flask server (usually run in main thread, or its own thread if needed)
        print(f"Laptop Main: Starting Flask status server on port {FLASK_PORT}...")
        # Run Flask in a separate thread so KeyboardInterrupt works in the main thread
        flask_thread = threading.Thread(target=lambda: laptop_app.run(host='0.0.0.0', port=FLASK_PORT, debug=False), daemon=True)
        flask_thread.start()

        # Keep main thread alive to catch KeyboardInterrupt
        while True:
            if not socket_thread.is_alive() or not flask_thread.is_alive():
                 print("Laptop Main: A required thread has stopped. Exiting.")
                 break
            time.sleep(1)


    except KeyboardInterrupt:
        print("\nLaptop Main: Ctrl+C pressed.")
    finally:
        print("Laptop Main: Setting stop flag...")
        stop_server_flag = True # Signal threads to stop

        if socket_thread and socket_thread.is_alive():
            print("Laptop Main: Waiting for socket thread...")
            socket_thread.join(timeout=3)
        if flask_thread and flask_thread.is_alive():
             print("Laptop Main: Flask thread is daemon, should exit.") # Flask might not join cleanly

        print("Laptop Main: Cleanup complete.")