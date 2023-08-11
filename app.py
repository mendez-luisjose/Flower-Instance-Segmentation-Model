import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from utils import set_background
from  streamlit_webrtc import webrtc_streamer
import av

set_background("./imgs/background.png")

LICENSE__FLOWER_MODEL = './model/flowers_segmentation_model.pt'

header = st.container()
body = st.container()

flower_model = YOLO(LICENSE__FLOWER_MODEL)

threshold = 0.30

state = "Uploader"

if "state" not in st.session_state :
    st.session_state["state"] = "Uploader"

class VideoProcessor:
    def recv(self, frame) :
        img = frame.to_ndarray(format="bgr24")
        pred = flower_model.predict(img)[0]
        img_wth_box = pred.plot()

        return av.VideoFrame.from_ndarray(img_wth_box, format="bgr24")
    
def model_prediction(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pred = flower_model.predict(img)[0]
    img_wth_box = pred.plot()
    img_wth_box = cv2.cvtColor(img_wth_box, cv2.COLOR_BGR2RGB)
    return img_wth_box

def change_state_uploader() :
    st.session_state["state"] = "Uploader"

    
def change_state_camera() :
    st.session_state["state"] = "Camera"

def change_state_live() :
    st.session_state["state"] = "Live"
    
with header :
    _, col1, _ = st.columns([0.25,1,0.1])
    col1.title("💥 Flower Instance Segmentation 🌻")

    video_file = open('./videos/video_footage.webm', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

    _, col4, _ = st.columns([0.1,1,0.2])
    col4.subheader("Computer Vision Detection with YoloV8 🧪")

    _, col5, _ = st.columns([0.05,1,0.1])
    col5.image("./imgs/train_batch2.jpg")

    st.write("This are the Flower Images that the Model was trained, during 10 epochs, using more than 4000 images.")

    _, col, _ = st.columns([0.2,1,0.2])
    col.header("Flowers Images Examples 🌹:")

    _, col2, _ = st.columns([0.1,1,0.2])
    col2.image("./imgs/val_batch0_pred.jpg")

    st.write("The Model was trained with the Yolov8 Architecture, for 10 epochs, using the Google Colab GPU, and with more than 4000 Images.")
    st.image("./imgs/results.png")

with body :
    _, col1, _ = st.columns([0.25,1,0.2])
    col1.subheader("Check It-out the Flower Instance Segmentation Model 🔎!")

    _, colb1, colb2, colb3 = st.columns([0.2, 0.7, 0.6, 1])

    if colb1.button("Upload an Image", on_click=change_state_uploader) :
        pass
    elif colb2.button("Take a Photo", on_click=change_state_camera) :
        pass
    elif colb3.button("Live Detection", on_click=change_state_live) :
        pass

    if st.session_state["state"] == "Uploader" :
        img = st.file_uploader("Upload a Flower Image: ", type=["png", "jpg", "jpeg"])
    elif st.session_state["state"] == "Camera" :
        img = st.camera_input("Take a Photo: ")
    elif st.session_state["state"] == "Live" :
        webrtc_streamer(key="sample", video_processor_factory=VideoProcessor)
        img = None

    _, col2, _ = st.columns([0.3,1,0.2])

    _, col5, _ = st.columns([0.8,1,0.2])

    if img is not None:
        image = np.array(Image.open(img))    
        col2.image(image, width=400)

        if col5.button("Apply Detection"):
            results = model_prediction(image)
            
            _, col3, _ = st.columns([0.4,1,0.2])
            col3.header("Detection Results ✅:")

            _, col4, _ = st.columns([0.1,1,0.1])
            col4.image(results)




 