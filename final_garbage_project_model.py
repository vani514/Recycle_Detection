import streamlit as st
import numpy as np
import av
from ultralytics import YOLO
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Load YOLO model
model = YOLO("final_model_garbage.pt")

class GarbageDetectionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO inference
        results = model(img)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]}: {conf:.2f}"

                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Garbage Classification Model")

webrtc_streamer(
    key="garbage-detection",
    video_processor_factory=GarbageDetectionProcessor
)
