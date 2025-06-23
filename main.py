import cv2  # OpenCV for video processing and visualization
from ultralytics import YOLO  # Ultralytics YOLOv8 for object detection
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT tracker for multi-object tracking

# --- CONFIGURATION ---
# Paths to the trained YOLO model, input video, and output file
model_path = r'C:\Users\PRJAWAL\OneDrive\Desktop\Assignment\best.pt'
video_path = r'C:\Users\PRJAWAL\OneDrive\Desktop\Assignment\15sec_input_720p.mp4'
output_path = r'C:\Users\PRJAWAL\OneDrive\Desktop\Assignment\output.mp4'

# --- Load YOLO model ---
model = YOLO(model_path)  # Load custom-trained YOLOv8 model for detecting players

# --- Load video ---
cap = cv2.VideoCapture(video_path)  # Open the input video file
if not cap.isOpened():
    print("❌ Error: Could not open video.")  # If video file can't be opened, exit the script
    exit()

# Get video properties: width, height, and frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# --- Output video writer ---
# Define codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# --- Initialize Deep SORT Tracker ---
tracker = DeepSort(max_age=30)  # Initialize DeepSORT tracker with max_age to keep tracks alive

# --- Main Loop ---
while cap.isOpened():  # Loop through video frames
    ret, frame = cap.read()  # Read next frame
    if not ret:
        print("✅ End of video.")  # End of video reached
        break

    # Run YOLO object detection on the frame
    results = model(frame, verbose=False)[0]  # Get results (boxes, scores, classes)
    detections = []  # List to hold detections for tracking

    # Loop through all detections
    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r.tolist()  # Unpack detection info
        
        # Only detect class 0 = player; ignore others like ball
        if int(cls) == 0 and conf > 0.4:  # Confidence threshold
            w, h = x2 - x1, y2 - y1  # Convert box from (x1,x2) to (x,y,w,h)
            bbox = [x1, y1, w, h]
            detections.append((bbox, conf, 'player'))  # Add to detection list

    # Update DeepSORT tracker with current frame's detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw bounding boxes and labels
    for track in tracks:
        if not track.is_confirmed():
            continue  # Skip unconfirmed tracks

        track_id = track.track_id  # Unique ID assigned by DeepSORT
        x1, y1, x2, y2 = map(int, track.to_ltrb())  # Get box in (left, top, right, bottom) format

        # Clamp coordinates to stay within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)

        # Draw bounding box and player ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Player {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write the frame to output video file
    out.write(frame)
    
    # Show the frame in a window (real-time visualization)
    cv2.imshow("Player Re-Identification", frame)

    # Quit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting early by user.")
        break

# --- Cleanup ---
cap.release()  # Release video capture object
out.release()  # Release video writer
cv2.destroyAllWindows()  # Close OpenCV display window
print("✅ Done! Output saved to:", output_path)
