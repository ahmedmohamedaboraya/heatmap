import streamlit as st
import tempfile
import os
import numpy as np
from io import BytesIO
import cv2
from ultralytics import YOLO
#model= YOLO("yolov8x.pt")

def process_video(video_path, output_path):    # Create VideoCapture from video paths
    kamera = cv2.VideoCapture(video_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:

        ret, kare = kamera.read()
        if not ret:
            break

        imgs = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        results = model(imgs, verbose=False)
        # burada her bir kare için noktalar sıfırlanır
        points = []
        labels = results[0].names
        for i in range(len(results[0].boxes)):
            x1, y1, x2, y2 = results[0].boxes.xyxy[i]
            score = results[0].boxes.conf[i]
            label = results[0].boxes.cls[i]
            x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
            name = labels[label]

            if score < 0.1:
                continue

            if name != 'person':
                continue

            center_x = int(x1 / 2 + x2 / 2)
            center_y = int(y1 / 2 + y2 / 2)
            points.append([center_x, center_y])

        points = np.array(points)
        heatmap, xedges, yedges = np.histogram2d(points[:, 1], points[:, 0], bins=15,
                                                 range=[[0, kare.shape[0]], [0, kare.shape[1]]])
        heatmap = cv2.resize(heatmap, (kare.shape[1], kare.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        result = cv2.addWeighted(kare, 0.6, heatmap, 0.8, 0)

        cv2.imshow('CIPU', result)

        cv2.imshow("kamera", kare)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    kamera.release()
    cv2.destroyAllWindows()
def main():
    # Streamlit app title    st.title("YOLOv8 Inference with Streamlit")
    # File uploader for video
    video_file = st.file_uploader("Choose a video file", type=["mp4"])
    if video_file:        # Save the uploaded video to the working directory
        uploaded_video_path = "uploaded_video.mp4"
        with open(uploaded_video_path, "wb") as video_output:
            video_output.write(video_file.read())
        # Process video and save the output
        output_video_path = "output_video.mp4"
        process_video(uploaded_video_path,output_video_path)
        # Display the processed video using Streamlit
        st.video(output_video_path, format="video/mp4")

if __name__ == "__main__" :
    main()