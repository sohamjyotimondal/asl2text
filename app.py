import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import google.generativeai as genai

# from dotenv import load_dotenv # Keep if you use .env
import os
import base64  # For handling image data from web
from flask import Flask, render_template, request, jsonify
import io
from PIL import Image  # For converting data URL to image
import traceback  # For printing detailed errors

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
# load_dotenv() # Uncomment if you use a .env file
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# WARNING: Hardcoding keys is insecure. Use environment variables for production.
GOOGLE_API_KEY = "AIzaSyDb6lcp_v7kp3ZZp1s16z89X6JP6Y_QL-0"  # Replace with your actual key or load safely
if not GOOGLE_API_KEY:
    print("Warning: Google API Key not found. Sentence generation disabled.")

MODEL_PATH = "model.tflite"
TRAIN_CSV_PATH = "train.csv"
EXAMPLE_PARQUET_PATH = "10042041.parquet"
ROWS_PER_FRAME = 543
PREDICTION_INTERVAL_SECONDS = 3
# --- New configuration for cooldown ---
RECOGNITION_COOLDOWN_SECONDS = 2  # Delay after a new word is recognized

# --- Global Variables / State ---
collected_frame_data = []
last_prediction_time = time.time()
current_sign_prediction = "Initializing..."
unique_signs_in_batch = []
xyz_skeleton = None
interpreter = None
pred_fn = None
SIGN2ORD = {}
ORD2SIGN = {}
gemini_model = None
mp_holistic_module = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
holistic_instance = None
# --- New global for cooldown state ---
cooldown_end_time = 0.0  # Time when the current cooldown period ends


# --- Initialization Functions ---
# (load_resources function remains the same as the previous correct version)
def load_resources():
    global xyz_skeleton, interpreter, pred_fn, SIGN2ORD, ORD2SIGN, gemini_model
    global holistic_instance

    print("Loading resources...")
    # Load skeleton
    try:
        xyz = pd.read_parquet(EXAMPLE_PARQUET_PATH, columns=["type", "landmark_index"])
        xyz_skeleton = (
            xyz.iloc[:ROWS_PER_FRAME][["type", "landmark_index"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        xyz_skeleton["row_id"] = xyz_skeleton.index
        print(f"Loaded skeleton structure with {len(xyz_skeleton)} definitions.")
        if len(xyz_skeleton) != ROWS_PER_FRAME:
            print(
                f"Warning: Skeleton definition count ({len(xyz_skeleton)}) != ROWS_PER_FRAME ({ROWS_PER_FRAME})."
            )
    except Exception as e:
        print(
            f"FATAL Error loading skeleton structure from '{EXAMPLE_PARQUET_PATH}': {e}"
        )
        traceback.print_exc()
        exit()

    # Load TFLite Model
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        pred_fn = interpreter.get_signature_runner("serving_default")
        print("TFLite model loaded successfully.")
    except Exception as e:
        print(f"FATAL Error loading TFLite model '{MODEL_PATH}': {e}")
        traceback.print_exc()
        exit()

    # Load Sign Mappings
    try:
        train = pd.read_csv(TRAIN_CSV_PATH)
        train["sign_ord"] = train["sign"].astype("category").cat.codes
        SIGN2ORD = train[["sign", "sign_ord"]].set_index("sign").squeeze().to_dict()
        ORD2SIGN = train[["sign_ord", "sign"]].set_index("sign_ord").squeeze().to_dict()
        print(f"Loaded {len(ORD2SIGN)} sign mappings.")
    except Exception as e:
        print(f"FATAL Error loading train data '{TRAIN_CSV_PATH}': {e}")
        traceback.print_exc()
        exit()

    # Initialize MediaPipe Holistic instance
    try:
        holistic_instance = mp_holistic_module.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        print("MediaPipe Holistic model instance initialized.")
    except Exception as e:
        print(f"FATAL Error initializing MediaPipe Holistic instance: {e}")
        traceback.print_exc()
        exit()

    # Initialize Gemini
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            print("Google Gemini model initialized.")
        except Exception as e:
            print(
                f"Warning: Error initializing Google Gemini: {e}. Sentence generation might fail."
            )
            traceback.print_exc()
            gemini_model = None
    else:
        print("Warning: Google API Key not provided. Sentence generation disabled.")
        gemini_model = None

    print("Resource loading complete.")


# --- Landmark Handling Functions ---
# (extract_landmarks_as_numpy function remains the same)
def extract_landmarks_as_numpy(results, xyz_skeleton):
    landmarks = []

    def add_landmarks(landmark_list, landmark_type):
        if landmark_list:
            for i, point in enumerate(landmark_list.landmark):
                landmarks.append(
                    {
                        "type": landmark_type,
                        "landmark_index": i,
                        "x": point.x,
                        "y": point.y,
                        "z": point.z,
                    }
                )

    add_landmarks(results.face_landmarks, "face")
    add_landmarks(results.pose_landmarks, "pose")
    add_landmarks(results.left_hand_landmarks, "left_hand")
    add_landmarks(results.right_hand_landmarks, "right_hand")

    if not landmarks:
        return np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)
    landmarks_df = pd.DataFrame(landmarks)
    merged_df = xyz_skeleton.merge(
        landmarks_df, on=["type", "landmark_index"], how="left"
    )
    merged_df = merged_df.sort_values(by="row_id")
    landmark_values = (
        merged_df[["x", "y", "z"]].fillna(np.nan).values.astype(np.float32)
    )
    if len(landmark_values) < ROWS_PER_FRAME:
        padding = np.full(
            (ROWS_PER_FRAME - len(landmark_values), 3), np.nan, dtype=np.float32
        )
        landmark_values = np.vstack((landmark_values, padding))
    elif len(landmark_values) > ROWS_PER_FRAME:
        landmark_values = landmark_values[:ROWS_PER_FRAME, :]
    return landmark_values


# --- API Call Function ---
# (get_display_message_from_api function remains the same)
def get_display_message_from_api(recognised_words):
    if not gemini_model:
        print("Gemini model not available. Returning raw words.")
        filtered_words = [word for word in recognised_words if word.lower() != "tv"]
        return (
            f"Recognized: {' '.join(filtered_words)}"
            if filtered_words
            else "No relevant words recognized."
        )
    if not recognised_words:
        return "No words recognized to form a sentence."
    filtered_words = [word for word in recognised_words if word.lower() != "tv"]
    if not filtered_words:
        return "No relevant words recognized (only 'TV' found)."
    prompt = f"Objective: You are given a list of recognized American Sign Language (ASL) words, possibly out of order. Construct a descriptive, coherent English sentence using these words If it is something like hello girl callonphone shh . IT means something like hey girl please keep quiet while on the phone. Ignore 'TV' and if'shhh' is there it means something like quiet. Make it grammatically correct and make it somwhat descriptive and long so people can understand well .\nInput List: {filtered_words}\nOutput Sentence:"
    try:
        response = gemini_model.generate_content(prompt)
        if response.parts:
            return response.text.strip()
        elif (
            hasattr(response, "prompt_feedback")
            and response.prompt_feedback.block_reason
        ):
            print(
                f"Gemini API blocked prompt. Reason: {response.prompt_feedback.block_reason}"
            )
            return f"Content blocked by API. Recognized: {' '.join(filtered_words)}"
        else:
            return f"API returned empty. Recognized: {' '.join(filtered_words)}"
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        traceback.print_exc()
        return f"API Error. Recognized: {' '.join(filtered_words)}"


# --- Flask Routes ---


@app.route("/")
def index():
    """Render the main HTML page."""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_frame():
    """Process a single frame received from the browser."""
    global collected_frame_data, last_prediction_time, current_sign_prediction, unique_signs_in_batch
    global cooldown_end_time  # Need to modify this global

    # Ensure resources are loaded
    if holistic_instance is None or pred_fn is None or xyz_skeleton is None:
        print("Error: Resources not loaded properly!")
        return jsonify({"error": "Server resources not initialized"}), 500

    # Get current time once for efficiency
    current_time = time.time()
    # Check if we are currently in a cooldown period
    is_in_cooldown = current_time < cooldown_end_time

    status_message = "Processing..."  # Default status

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data received"}), 400

        # Decode Base64 image data
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image is None:
            print("Error: Could not decode image from base64 data.")
            return jsonify({"error": "Could not decode image"}), 400

        # --- MediaPipe Processing (Always run for live video feed) ---
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic_instance.process(image_rgb)
        image_rgb.flags.writeable = True

        # --- Landmark Extraction (Always run) ---
        landmarks_np = extract_landmarks_as_numpy(results, xyz_skeleton)

        # --- Collect Data ONLY IF NOT IN COOLDOWN ---
        if not is_in_cooldown:
            if not np.all(np.isnan(landmarks_np)):
                collected_frame_data.append(landmarks_np)
                status_message = f"Collected {len(collected_frame_data)} frames..."
            else:
                status_message = "No landmarks detected in frame."
        else:
            # If in cooldown, update status message to indicate it
            remaining_cooldown = cooldown_end_time - current_time
            status_message = f"Cooldown active... {remaining_cooldown:.1f}s left"

        # --- Prediction Logic (Interval-based AND NOT IN COOLDOWN) ---
        # Only run prediction check if enough time has passed AND we are not in cooldown
        if (
            not is_in_cooldown
            and current_time - last_prediction_time >= PREDICTION_INTERVAL_SECONDS
        ):
            status_message = f"Time interval reached ({PREDICTION_INTERVAL_SECONDS}s)."
            if collected_frame_data:
                input_data = np.stack(collected_frame_data, axis=0).astype(np.float32)
                # input_data = np.nan_to_num(input_data, nan=0.0) # If model needs 0s

                new_word_recognized_this_cycle = (
                    False  # Flag to check if cooldown needs starting
                )
                try:
                    prediction = pred_fn(inputs=input_data)
                    sign_ord = prediction["outputs"].argmax()
                    sign_name = ORD2SIGN.get(sign_ord, "Unknown Sign")
                    current_sign_prediction = (
                        sign_name  # Update prediction display immediately
                    )

                    # Check if it's a new, valid sign to add to the batch
                    if sign_name != "TV" and sign_name != "Unknown Sign":
                        if sign_name not in unique_signs_in_batch:
                            unique_signs_in_batch.append(sign_name)
                            # --- Trigger Cooldown ---
                            print(
                                f"New word recognized: {sign_name}. Starting {RECOGNITION_COOLDOWN_SECONDS}s cooldown."
                            )
                            cooldown_end_time = (
                                current_time + RECOGNITION_COOLDOWN_SECONDS
                            )
                            new_word_recognized_this_cycle = (
                                True  # Mark that cooldown started
                            )
                            status_message = (
                                f"Recognized: {sign_name}! Cooldown started."
                            )
                        # else: sign already in batch, no cooldown needed
                    # Update status if no new word triggered cooldown message
                    if not new_word_recognized_this_cycle:
                        status_message = f"Prediction made: {current_sign_prediction}"

                except Exception as e:
                    print(f"Error during prediction: {e}")
                    traceback.print_exc()
                    current_sign_prediction = "Prediction Error"
                    status_message = f"Prediction Error: {e}"

                # Reset collected data and prediction timer after processing this interval
                collected_frame_data = []
                last_prediction_time = current_time
            else:
                # No frames collected in interval, update prediction display
                current_sign_prediction = "No Movement Detected"
                status_message = "No frames collected in interval."
                # Reset prediction timer even if no data was collected
                last_prediction_time = current_time
        # --- End of Prediction Logic Block ---

        # --- Drawing Landmarks (Always run) ---
        image_bgr_annotated = image.copy()
        mp_drawing.draw_landmarks(
            image_bgr_annotated,
            results.face_landmarks,
            mp_holistic_module.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp_drawing.draw_landmarks(
            image_bgr_annotated,
            results.pose_landmarks,
            mp_holistic_module.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        mp_drawing.draw_landmarks(
            image_bgr_annotated,
            results.left_hand_landmarks,
            mp_holistic_module.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
        )
        mp_drawing.draw_landmarks(
            image_bgr_annotated,
            results.right_hand_landmarks,
            mp_holistic_module.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
        )

        # --- Encode image (Always run) ---
        retval, buffer = cv2.imencode(".jpg", image_bgr_annotated)
        if not retval:
            print("Error: Could not encode annotated image to JPEG.")
            return (
                jsonify(
                    {
                        "error": "Could not encode annotated image",
                        "prediction": current_sign_prediction,
                        "words": unique_signs_in_batch,
                        "status": status_message,
                        "image": None,
                    }
                ),
                500,
            )

        encoded_image = base64.b64encode(buffer).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{encoded_image}"

        # --- Return results (always includes latest image, potentially stale prediction/words during cooldown) ---
        return jsonify(
            {
                "image": data_url,
                "prediction": current_sign_prediction,  # This reflects the latest prediction made (could be stale during cooldown)
                "words": unique_signs_in_batch,  # These are words collected before the current cooldown started
                "status": status_message,  # Updated status reflects cooldown state
            }
        )

    except Exception as e:
        # General catch-all
        print(f"Error in /process: {e}")
        traceback.print_exc()
        # Return error state, preserving last known good prediction/words if possible
        return (
            jsonify(
                {
                    "error": f"Server processing error: {e}",
                    "prediction": current_sign_prediction,
                    "words": unique_signs_in_batch,
                    "status": f"Error: {e}",
                    "image": None,
                }
            ),
            500,
        )


@app.route("/generate", methods=["POST"])
def generate_sentence():
    """Call the Gemini API to generate a sentence from collected words."""
    global unique_signs_in_batch
    print(f"Generating sentence from: {unique_signs_in_batch}")
    status_message = ""

    if not unique_signs_in_batch:
        sentence = "No words were collected in the last batch to generate a sentence."
        status_message = "No words to process."
        return jsonify({"sentence": sentence, "status": status_message})

    try:
        words_to_process = list(unique_signs_in_batch)
        sentence = get_display_message_from_api(words_to_process)
        unique_signs_in_batch = []  # Clear batch after processing attempt
        print(f"Generated sentence attempt result: {sentence}")
        status_message = "Sentence processed."
        return jsonify({"sentence": sentence, "status": status_message})

    except Exception as e:
        print(f"Error in /generate: {e}")
        traceback.print_exc()
        status_message = f"Sentence generation error: {e}"
        # Don't clear unique_signs_in_batch if API call failed
        return (
            jsonify(
                {
                    "sentence": f"Error generating sentence: {e}",
                    "status": status_message,
                }
            ),
            500,
        )


# --- Main Execution ---
if __name__ == "__main__":
    load_resources()
    print("--- Server Ready ---")
    print(f"Recognition cooldown set to: {RECOGNITION_COOLDOWN_SECONDS} seconds")
    print("Access the application:")
    print(" - On this computer: http://127.0.0.1:3000 or http://localhost:3000")
    print(" - On other devices on the same network: http://<YOUR_COMPUTER_IP>:3000")
    print("   (Find IP using 'ipconfig' on Win or 'ip addr'/'ifconfig' on Mac/Linux)")
    print("Press Ctrl+C to stop the server.")
    app.run(host="0.0.0.0", port=3000, debug=True, threaded=True, use_reloader=False)
