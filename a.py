import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import time

def load_model():
    model = YOLO(r"/home/rguktongole/Desktop/garbage/detect/train/weights/best.pt")  # Load YOLO model
    return model

def detect_objects(model, image):
    results = model(image)
    return results

def draw_boxes(image, results):
    image = np.array(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            conf = box.conf[0].item()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 6)  # Increased bounding box thickness
            cv2.putText(image, f'{label}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)  # Increased font thickness
    return Image.fromarray(image)

def get_disposal_methods():
    return {
        "Aerosol": "🛢️ Dispose at a hazardous waste facility.",
        "Aluminium blister pack": "♻️ Recycle if clean; otherwise, dispose in general waste.",
        "Aluminium foil": "♻️ Recycle if free of food residue.",
        "Battery": "⚡ Drop off at an e-waste or battery recycling center.",
        "Broken glass": "🚨 Wrap securely and dispose in general waste.",
        "Carded blister pack": "♻️ Recycle if possible, otherwise general waste.",
        "Cigarette": "🚬 Dispose in a designated cigarette waste bin.",
        "Clear plastic bottle": "♻️ Recycle in plastic recycling bins.",
        "Corrugated carton": "📦 Recycle in paper recycling.",
        "Crisp packet": "🚮 Dispose in general waste or TerraCycle if available.",
        "Disposable food container": "♻️ Recycle if clean, otherwise general waste.",
        "Disposable plastic cup": "♻️ Recycle in plastic recycling bins.",
        "Drink can": "♻️ Recycle in metal recycling.",
        "Drink carton": "♻️ Recycle in carton recycling bins.",
        "Egg carton": "🌱 Compost or recycle in paper recycling.",
        "Foam cup": "🚯 Dispose in general waste.",
        "Foam food container": "🚯 Dispose in general waste.",
        "Food Can": "♻️ Recycle in metal recycling bins.",
        "Food waste": "🌱 Compost or dispose in biodegradable waste.",
        "Garbage bag": "🚮 Dispose in general waste.",
        "Glass bottle": "🟢 Recycle in glass recycling bins.",
        "Glass cup": "🚯 Dispose in general waste if broken.",
        "Glass jar": "🟢 Recycle in glass recycling bins.",
        "Magazine paper": "📄 Recycle in paper recycling bins.",
        "Meal carton": "♻️ Recycle in paper or carton recycling bins.",
        "Metal bottle cap": "♻️ Recycle in metal recycling bins.",
        "Metal lid": "♻️ Recycle in metal recycling bins.",
        "Normal paper": "📄 Recycle in paper recycling bins.",
        "Other carton": "♻️ Recycle in carton recycling bins.",
        "Other plastic": "🚮 Dispose in general waste if not recyclable.",
        "Other plastic bottle": "♻️ Recycle in plastic recycling bins.",
        "Other plastic container": "♻️ Recycle in plastic recycling bins.",
        "Other plastic cup": "♻️ Recycle in plastic recycling bins.",
        "Other plastic wrapper": "🚯 Dispose in general waste.",
        "Paper bag": "📄 Recycle in paper recycling bins.",
        "Paper cup": "♻️ Recycle if lined with recyclable plastic.",
        "Paper straw": "🌱 Compost or dispose in paper recycling.",
        "Pizza box": "🚯 Dispose in general waste if greasy, otherwise recycle.",
        "Plastic bottle cap": "♻️ Recycle in plastic recycling bins.",
        "Plastic film": "🔄 Dispose in general waste unless recyclable locally.",
        "Plastic gloves": "🚯 Dispose in general waste.",
        "Plastic lid": "♻️ Recycle in plastic recycling bins.",
        "Plastic straw": "🚯 Dispose in general waste.",
        "Plastic utensils": "🚯 Dispose in general waste.",
        "Polypropylene bag": "♻️ Recycle at designated plastic bag drop-offs.",
        "Pop tab": "♻️ Recycle in metal recycling bins.",
        "Rope - strings": "🚯 Dispose in general waste.",
        "Scrap metal": "🔧 Recycle at a scrap metal collection center.",
        "Shoe": "👞 Donate if in good condition, otherwise dispose in general waste.",
        "Single-use carrier bag": "♻️ Recycle at designated plastic bag drop-offs.",
        "Six pack rings": "🚯 Dispose in general waste after cutting loops.",
        "Spread tub": "♻️ Recycle in plastic recycling bins.",
        "Squeezable tube": "🚯 Dispose in general waste.",
        "Styrofoam piece": "🚯 Dispose in general waste.",
        "Tissues": "🗑️ Dispose in general waste.",
        "Toilet tube": "🌱 Compost or recycle in paper recycling.",
        "Tupperware": "♻️ Recycle if plastic type is accepted, otherwise general waste.",
        "Unlabeled litter": "🚯 Dispose in general waste.",
        "Wrapping paper": "🎁 Recycle if non-metallic, otherwise general waste."
    }

def home_page():
    st.title("♻️ Recycle Detection:Intelligent Recyclable Recognition")
    st.write("An ML based waste classification system that detects waste types and provides recycle recommendations.")
    if st.button("Get Started 🚀"):
        st.session_state.page = "upload"
        # st.experimental_rerun()

def upload_page():
    st.title("♻️ Recycle Detection:Intelligent Recyclable Recognition")
    st.header("📤 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        model = load_model()
        results = detect_objects(model, image)
        processed_image = draw_boxes(image, results)
        
        st.image(processed_image, caption="Processed Image", width=300)
        
        disposal_methods = get_disposal_methods()
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                st.write(f"**{label}** - {disposal_methods.get(label, 'No recycling method found.')}")

def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    if st.session_state.page == "home":
        home_page()
    else:
        upload_page()

if __name__ == "__main__":
    main()