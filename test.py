from ultralytics import YOLO
from PIL import Image
import cv2
import os

# Load the YOLO model
model = YOLO('/home/codezeros/Documents/Fast_API/best.pt')

# Open the video file
cap = cv2.VideoCapture('/home/codezeros/Documents/Fast_API/4440932-hd_1920_1080_25fps.mp4')
# Define the output directory
output_video_dir = 'results/video'
output_frames_dir = 'results/frames'
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_frames_dir, exist_ok=True)

# Get the frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_video_dir, 'output_video.mp4')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))


label = "smoke"

window_name = 'Frame'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Adjust the frame size to make it larger
desired_width = 1280
desired_height = 720
cv2.resizeWindow(window_name, desired_width, desired_height)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    ## Display or save the frame with detection results
    for result in results:
        for box in result.boxes.xyxy:
            conf = result.boxes.conf[0]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
            # cv2.putText(frame, f"{label}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(output_frames_dir, f"detected_frame_{conf:.2f}.jpg"), frame)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    # cv2.imwrite(os.path.join(output_frames_dir, f"detected_frame_{conf:.2f}.jpg"), frame)
   
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Output video saved at: {output_video_path}")
