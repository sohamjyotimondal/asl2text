# Sign Language to Coherent Sentences: Giving a Voice to the Mute

## Overview

This project presents a real-time American Sign Language (ASL) recognition and translation system, running entirely on a Raspberry Pi 4B with a 5 MP camera module. The system captures video, extracts human keypoints using Mediapipe, classifies the performed sign using a custom Transformer-based deep learning model (converted to TFLite), and composes meaningful sentences using Gemini 2.0 Flash—all on-device.

---

## Problem Statement

Communication barriers persist between the Deaf community and non-signers due to a lack of affordable, accessible, and portable ASL translation solutions. This project addresses the challenge by delivering a real-time, low-latency, and efficient ASL recognition and translation pipeline deployable on resource-constrained hardware.

---

## Components Used

- **Raspberry Pi 4B**
- **Raspberry Pi Camera Module Rev 1.3**

---

## Data Acquisition

The dataset is sourced from the [Kaggle American Sign Language Recognition Competition](https://www.kaggle.com/competitions/asl-signs). It contains video sequences of signers performing ASL gestures, with each frame annotated by 543 body landmarks (x, y, z) using Mediapipe. The dataset covers 250 sign classes and provides a diverse set of signers for robust model training.

**Preprocessing Steps:**
- **Frame Extraction:** Video is segmented into 3-second windows, with a 2-second cooldown between captures.
- **Landmark Extraction:** Each frame is processed using Mediapipe to extract 543 keypoints.
- **Normalization:** Landmarks are normalized; missing data is handled with masking.
- **Label Encoding:** Sign labels are mapped to integer codes for training.

---

## Model Architecture

- **Landmark Embedding:** Separate dense layers for lips, left hand, and pose regions. Missing landmarks use a learned "empty" embedding.
- **Positional Encoding:** Temporal (frame) position information is added.
- **Transformer Encoder:** Multi-head self-attention blocks model temporal dependencies across frames.
- **Pooling & Classification:** Sequence is mean-pooled and passed to a softmax classifier.
- **Regularization:** Label smoothing, dropout, and weight decay for improved generalization.

The model is trained on TensorFlow and converted to TensorFlow Lite for efficient, on-device inference.

---

## Inference Pipeline

1. **Capture:** 3 seconds of video frames via the Pi camera module.
2. **Preprocess:** Extract and normalize landmarks with Mediapipe.
3. **Classify:** Use TFLite model to predict the ASL sign for each window.
4. **Aggregate:** Store recognized signs in a word array.
5. **Compose Sentences:** Use Gemini 2.0 Flash LLM to convert recognized words into coherent sentences.
6. **Display:** Output the final sentence to the user.

---

## How to Run

1. **Hardware Setup:** Connect the Raspberry Pi 4B and attach the Camera Module Rev 1.3.
2. **Install Dependencies:**
   - Python 3.9
   - TensorFlow Lite runtime
   - Mediapipe
   - Gemini 2.0 Flash (or other supported LLM)
3. **Dataset Preparation:** Download and preprocess the Kaggle ASL dataset as described above.
4. **Model Training:** Train the model using the provided scripts/notebooks or use a pretrained model.
5. **Model Conversion:** Convert the trained model to TFLite format.
6. **Deploy and Run:** Deploy the inference scripts and models on the Pi. Run the app.py script to start real-time ASL recognition and translation.


---

## References

- [Kaggle ASL Signs Competition](https://www.kaggle.com/competitions/asl-signs)
- [Mediapipe Documentation](https://ai.google.dev/edge/mediapipe/solutions/guide)

---

## Conclusion

This project bridges the communication gap for the Deaf community by providing a compact, real-time ASL translation system that is both affordable and portable. By combining computer vision, deep learning, and natural language processing on embedded hardware, it demonstrates a scalable and deployable solution for accessible human-computer interaction.
