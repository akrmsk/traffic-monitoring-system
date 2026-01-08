import streamlit as st
import cv2
import torch
import numpy as np
import time
from collections import deque
import pandas as pd


# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Traffic Monitor", layout="wide")
st.title("🚦 Real-Time Traffic Analytics Dashboard")


# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


model = load_model()

# --- APP LAYOUT ---
col1, col2, col3, col4 = st.columns(4)
car_metric = col1.empty()
motor_metric = col2.empty()
truck_metric = col3.empty()
fps_metric = col4.empty()

v_col, d_col = st.columns([2, 1])
video_placeholder = v_col.empty()
chart_placeholder = d_col.empty()

# --- INITIALIZE LOGIC (Same as your main script) ---
VIDEO_SOURCE = "traffic.mp4"
TARGET_CLASSES = ['car', 'motorcycle', 'bus', 'truck']
CONFIDENCE_THRESHOLD = 0.4
TRACKER_DISTANCE_THRESHOLD = 120

cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
COUNTING_LINE_Y = frame_height // 2

vehicle_counts = {class_name: 0 for class_name in TARGET_CLASSES}
counted_object_ids = deque(maxlen=200)
tracked_objects = {}
next_object_id = 0
start_time = time.time()
frame_count = 0

# --- PROCESSING LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Inference
    results = model(frame)
    detections = results.pandas().xyxy[0]

    current_detections = []
    for index, row in detections.iterrows():
        if row['name'] in TARGET_CLASSES and row['confidence'] > CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            current_detections.append(((xmin, ymin, xmax, ymax), row['name']))

    # Tracking Logic
    unmatched_detections = list(range(len(current_detections)))
    updated_tracked_objects = {}

    for obj_id, (last_pos, class_name) in tracked_objects.items():
        best_match_idx = -1
        min_dist = float('inf')
        for i in unmatched_detections:
            box, _ = current_detections[i]
            center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            dist = np.sqrt((center_x - last_pos[0]) ** 2 + (center_y - last_pos[1]) ** 2)
            if dist < TRACKER_DISTANCE_THRESHOLD and dist < min_dist:
                min_dist = dist
                best_match_idx = i

        if best_match_idx != -1:
            box, new_class_name = current_detections[best_match_idx]
            center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            updated_tracked_objects[obj_id] = ((center_x, center_y), new_class_name)

            # Line Crossing Logic
            if (last_pos[1] < COUNTING_LINE_Y <= center_y or last_pos[
                1] > COUNTING_LINE_Y >= center_y) and obj_id not in counted_object_ids:
                vehicle_counts[new_class_name] += 1
                counted_object_ids.append(obj_id)
            unmatched_detections.remove(best_match_idx)

    for i in unmatched_detections:
        box, class_name = current_detections[i]
        updated_tracked_objects[next_object_id] = (((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), class_name)
        next_object_id += 1
    tracked_objects = updated_tracked_objects

    # --- DRAWING ---
    cv2.line(frame, (0, COUNTING_LINE_Y), (frame_width, COUNTING_LINE_Y), (0, 255, 0), 2)

    # Calculate FPS
    fps_value = frame_count / (time.time() - start_time)
    # Black FPS Text as requested
    cv2.putText(frame, f"FPS: {fps_value:.2f}", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    for box, class_name in current_detections:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    # --- UPDATE STREAMLIT UI ---
    # Metrics
    car_metric.metric("Cars", vehicle_counts['car'])
    motor_metric.metric("Motorcycles", vehicle_counts['motorcycle'])
    truck_metric.metric("Trucks", vehicle_counts['truck'])
    fps_metric.metric("System Speed", f"{fps_value:.2f} FPS")

    # Chart
    df = pd.DataFrame(list(vehicle_counts.items()), columns=['Vehicle', 'Count'])
    chart_placeholder.bar_chart(df.set_index('Vehicle'))

    # Video (Convert BGR to RGB for Streamlit)
    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

cap.release()