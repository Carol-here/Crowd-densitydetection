import cv2
import numpy as np
import tensorflow as tf
import math
from collections import deque

# -----------------------------
# Load CNN model
# -----------------------------
model = tf.keras.models.load_model("best_model.keras")

labels = ["LOW","MEDIUM","HIGH"]

# -----------------------------
# Processing parameters
# -----------------------------
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360

AREA_THRESHOLD = 1200
MAX_ROI = 5
DISTANCE_THRESHOLD = 120

TREND_WINDOW = 20
SMOOTH_WINDOW = 10
STABILITY_FRAMES = 6
TREND_THRESHOLD = 0.8

density_history = deque(maxlen=TREND_WINDOW)
smooth_buffer = deque(maxlen=SMOOTH_WINDOW)

stable_density = 0
stable_count = 0

# -----------------------------
# Confidence calculation
# -----------------------------
def compute_confidence(probs):

    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = math.log(len(probs))

    return 1 - (entropy / max_entropy)

# -----------------------------
# Background subtraction
# -----------------------------
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

# -----------------------------
# Video
# -----------------------------
cap = cv2.VideoCapture(0)

previous_prediction = np.array([0.33,0.33,0.33])

# -----------------------------
# Main loop
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame,(PROCESS_WIDTH,PROCESS_HEIGHT))

    fg_mask = backSub.apply(frame_small)

    kernel = np.ones((5,5),np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

    contours,_ = cv2.findContours(fg_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    centers = []
    roi_predictions = []

    # -----------------------------
    # ROI detection
    # -----------------------------
    for contour in contours:

        area = cv2.contourArea(contour)

        if area < AREA_THRESHOLD:
            continue

        x,y,w,h = cv2.boundingRect(contour)

        aspect_ratio = w/float(h)

        if aspect_ratio < 0.2 or aspect_ratio > 1.5:
            continue

        boxes.append([x,y,x+w,y+h])
        centers.append((x+w//2,y+h//2))

        roi = frame_small[y:y+h,x:x+w]

        roi = cv2.resize(roi,(128,128))
        roi = roi / 255.0
        roi = np.expand_dims(roi,axis=0)

        prediction = model.predict(roi,verbose=0)[0]

        roi_predictions.append(prediction)

        if len(roi_predictions) >= MAX_ROI:
            break

    # -----------------------------
    # CNN aggregation
    # -----------------------------
    if len(roi_predictions) > 0:
        avg_prediction = np.mean(roi_predictions,axis=0)
    else:
        avg_prediction = previous_prediction

    confidence = compute_confidence(avg_prediction)

    stabilized_prediction = (confidence * avg_prediction) + ((1-confidence)*previous_prediction)

    previous_prediction = stabilized_prediction

    cnn_density = np.argmax(stabilized_prediction)

    # -----------------------------
    # Distance-based clustering
    # -----------------------------
    crowd_pairs = 0

    for i in range(len(centers)):
        for j in range(i+1,len(centers)):

            d = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))

            if d < DISTANCE_THRESHOLD:
                crowd_pairs += 1

    if crowd_pairs <= 1:
        spatial_density = 0
    elif crowd_pairs <= 4:
        spatial_density = 1
    else:
        spatial_density = 2

    # -----------------------------
    # Hybrid density fusion
    # -----------------------------
    final_density = round(0.6*spatial_density + 0.4*cnn_density)

    # -----------------------------
    # Temporal smoothing
    # -----------------------------
    smooth_buffer.append(final_density)
    smoothed_density = int(round(np.mean(smooth_buffer)))

    density_history.append(smoothed_density)

    # -----------------------------
    # Hysteresis stability filter
    # -----------------------------

    if smoothed_density == stable_density:
        stable_count = 0
    else:
        stable_count += 1

        if stable_count >= STABILITY_FRAMES:
            stable_density = smoothed_density
            stable_count = 0

    # -----------------------------
    # Trend prediction
    # -----------------------------
    trend = "STABLE"

    if len(density_history) == TREND_WINDOW:

        start = np.mean(list(density_history)[:10])
        end = np.mean(list(density_history)[-10:])

        if end - start > TREND_THRESHOLD:
            trend = "CROWD FORMING"

        elif start - end > TREND_THRESHOLD:
            trend = "CROWD DISPERSING"

    # -----------------------------
    # Draw bounding boxes
    # -----------------------------
    scale_x = frame.shape[1] / PROCESS_WIDTH
    scale_y = frame.shape[0] / PROCESS_HEIGHT

    for box in boxes:

        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # -----------------------------
    # Display info
    # -----------------------------
    cv2.putText(frame,
                f"Density: {labels[stable_density]}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)

    cv2.putText(frame,
                f"Confidence: {confidence:.2f}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,0),
                2)

    cv2.putText(frame,
                f"Trend: {trend}",
                (20,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2)

    cv2.imshow("Crowd Formation Monitor",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()