from ultralytics import YOLO
from roboflow import Roboflow

# Download the data using Roboflow API key
rf = Roboflow(api_key="your_api_key")
project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
version = project.version(14)
dataset = version.download("yolov11")

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/rock-paper-scissors-14/data.yaml", epochs=100, imgsz=640)
