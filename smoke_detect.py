import numpy as np
import os
import time
import cv2
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI()

model = YOLO("/home/codezeros/Documents/Fast_API/best.pt")

# Define the output directories
output_video_dir = 'Result/video'
output_frames_dir = 'Result/frame'
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_frames_dir, exist_ok=True)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_video_dir, 'output_video.mp4')

@app.post("/detect_smoke")
async def detect_smoke(file: UploadFile = File(...)):
    video_bytes = await file.read()

    # Save the uploaded video to a temporary file
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as temp_video_file:
        temp_video_file.write(video_bytes)

    # Open the video file
    cap = cv2.VideoCapture(temp_video_path)

    # Get the frame dimensions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

    label = "smoke"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        results = model(frame)

        # Display or save the frame with detection results
        for result in results:
            for box in result.boxes.xyxy:
                conf = result.boxes.conf[0]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imwrite(os.path.join(output_frames_dir, f"detected_frame_{conf:.2f}.jpg"), frame)

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    os.remove(temp_video_path)

    return JSONResponse(content={"message": f"Output video saved at: {output_video_path}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.75.190", port=8000)
