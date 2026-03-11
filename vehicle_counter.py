import time  # Library for measuring time (for FPS)
from collections import deque

import cv2
import numpy as np
import torch

# --- CONFIGURATION ---
VIDEO_SOURCE = "traffic.mp4"
OUTPUT_FILENAME = "runs/detect/traffic_fps_benchmark.mp4"
LINE_THICKNESS = 2
FONT_SCALE = 0.8
FONT_THICKNESS = 2
TARGET_CLASSES = ["car", "motorcycle", "bus", "truck"]
CONFIDENCE_THRESHOLD = 0.4
TRACKER_DISTANCE_THRESHOLD = 120
COUNTED_ID_MEMORY = 200


# --- MAIN SCRIPT ---
def main():
    print("Loading YOLOv5 model ('yolov5s')...")
    # This line loads the 'small' yolov5s model. You can change 'yolov5s' to 'yolov5m' for the optimization experiment.
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    print("Model loaded successfully.")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_SOURCE}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # --- HORIZONTAL LINE LOGIC ---
    COUNTING_LINE_Y = frame_height // 2
    print(f"Video Resolution: {frame_width}x{frame_height}. Horizontal counting line placed at y={COUNTING_LINE_Y}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (frame_width, frame_height))

    vehicle_counts = {class_name: 0 for class_name in TARGET_CLASSES}
    total_vehicle_count = 0
    counted_object_ids = deque(maxlen=COUNTED_ID_MEMORY)
    tracked_objects = {}
    next_object_id = 0

    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    print("Starting video processing and performance benchmarking...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame)
        detections = results.pandas().xyxy[0]

        current_detections = []
        for index, row in detections.iterrows():
            if row["name"] in TARGET_CLASSES and row["confidence"] > CONFIDENCE_THRESHOLD:
                xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
                class_name = row["name"]
                current_detections.append(((xmin, ymin, xmax, ymax), class_name))

        unmatched_detections = list(range(len(current_detections)))
        updated_tracked_objects = {}

        for obj_id, (last_pos, class_name) in tracked_objects.items():
            best_match_idx = -1
            min_dist = float("inf")
            for i in unmatched_detections:
                box, _ = current_detections[i]
                center_x, center_y = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                dist = np.sqrt((center_x - last_pos[0]) ** 2 + (center_y - last_pos[1]) ** 2)
                if dist < TRACKER_DISTANCE_THRESHOLD:
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = i

            if best_match_idx != -1:
                box, new_class_name = current_detections[best_match_idx]
                center_x, center_y = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                updated_tracked_objects[obj_id] = ((center_x, center_y), new_class_name)
                if (
                    last_pos[1] < COUNTING_LINE_Y <= center_y or last_pos[1] > COUNTING_LINE_Y >= center_y
                ) and obj_id not in counted_object_ids:
                    vehicle_counts[new_class_name] += 1
                    total_vehicle_count += 1
                    counted_object_ids.append(obj_id)
                if best_match_idx in unmatched_detections:
                    unmatched_detections.remove(best_match_idx)

        for i in unmatched_detections:
            box, class_name = current_detections[i]
            center_x, center_y = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            updated_tracked_objects[next_object_id] = ((center_x, center_y), class_name)
            next_object_id += 1
        tracked_objects = updated_tracked_objects

        # --- VISUALIZATION ---
        cv2.line(
            frame, (0, COUNTING_LINE_Y), (frame_width, COUNTING_LINE_Y), (0, 255, 0), LINE_THICKNESS
        )  # Green horizontal line

        y_offset = 30
        for class_name, count in vehicle_counts.items():
            count_text = f"{class_name.capitalize()}: {count}"
            cv2.putText(
                frame, count_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS
            )
            y_offset += 30

        total_text = f"Total Vehicles: {total_vehicle_count}"
        cv2.putText(
            frame, total_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS
        )

        # Calculate and display FPS
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps_value = 0
        if elapsed_time > 0:
            fps_value = frame_count / elapsed_time
        fps_text = f"FPS: {fps_value:.2f}"

        # Change color from (255, 255, 255) to (0, 0, 0) for black text
        cv2.putText(
            frame, fps_text, (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS
        )

        for box, class_name in current_detections:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Blue detection boxes
            cv2.putText(frame, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow("Vehicle Counter Benchmark", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Benchmarking finished.")
    final_avg_fps = 0
    if time.time() - start_time > 0:
        final_avg_fps = frame_count / (time.time() - start_time)
    print(f"Average FPS: {final_avg_fps:.2f}")


if __name__ == "__main__":
    main()
