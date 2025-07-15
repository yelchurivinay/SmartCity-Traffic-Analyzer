import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLOv8 model (default small model)
model = YOLO("yolov8n.pt")

# Traffic detection function
def detect_traffic(image):
    img = np.array(image)
    results = model.predict(img)
    
    # YOLO annotated image
    annotated_img = results[0].plot()
    
    # Vehicle counting
    classes = results[0].names
    counts = {}
    for cls_id in results[0].boxes.cls:
        cls_name = classes[int(cls_id)]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    # Heatmap generation
    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        heatmap[y1:y2, x1:x2] += 1
    heatmap_img = cv2.applyColorMap((heatmap * 50).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 0.7, heatmap_img, 0.3, 0)

    return Image.fromarray(annotated_img), Image.fromarray(blended), counts

# Build Gradio app
with gr.Blocks() as demo:
    gr.Markdown("## 🚦 SmartCity Traffic Analyzer")
    with gr.Tab("Traffic Detection"):
        image_input = gr.Image(type="pil", label="Upload Traffic Image")
        with gr.Row():
            yolo_output = gr.Image(label="YOLO Detection")
            heatmap_output = gr.Image(label="Heatmap")
        result_output = gr.Label(label="Vehicle Count")
        btn = gr.Button("Analyze")
        btn.click(fn=detect_traffic, inputs=image_input, outputs=[yolo_output, heatmap_output, result_output])
    
    with gr.Tab("Route Optimization"):
        gr.Markdown("Coming Soon: Shortest Route Prediction")

demo.launch()
