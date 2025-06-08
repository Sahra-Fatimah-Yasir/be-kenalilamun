from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import torch
import json
import numpy as np
import io
from fastapi.staticfiles import StaticFiles
from torchvision.transforms import functional as F
from torchvision.transforms import Resize
# =====================================
# 🚀 INIT FASTAPI APP
# =====================================
app = FastAPI()
app.mount("/gambar_lamun", StaticFiles(directory="gambar_lamun"), name="gambar_lamun")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# 📚 LABEL MAPPING
# =====================================
COCO_CLASSES = {
    0: "Background",
    1: "Cymodocea_rotundata",
    2: "Enhalus_acoroides",
    3: "Syringodium_isoetifolium",
    4: "Thalassia_hemprichii"
}

# =====================================
# 🔧 LOAD MODEL
# =====================================
def load_model(weights_path, num_classes=5, device="cpu"):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load model saat startup
MODEL_PATH = "best_model.pth"  
model = load_model(MODEL_PATH)
device = "cpu"

# =====================================
# 🔍 LOAD JSON DATA
# =====================================
with open("dataTanaman.json") as f:
    tanaman_data = json.load(f)

def get_tanaman_by_label(label_name):
    for data in tanaman_data:
        if label_name.lower().replace(" ", "-") in data["nama"].lower().replace(" ", "-"):
            return data
    return None

# =====================================
# 🛬 ROUTES
# =====================================
@app.get("/")
def root():
    return {"message": "FastAPI backend for lamun Faster R-CNN ready!"}

@app.get("/lamun/get-data")
def get_data():
    return JSONResponse(content=tanaman_data)

@app.post("/lamun/detect")
async def detect_image(file: UploadFile = File(...), threshold: float = 0.2):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize ke 640x640 agar cocok dengan training model
    image_resized = image.resize((640, 640))

    # Konversi ke tensor
    image_tensor = F.to_tensor(image_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    results = []
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score >= threshold:
            class_name = COCO_CLASSES.get(label.item(), f"Class {label.item()}")
            x1, y1, x2, y2 = map(float, box.tolist())

            # Skala bounding box dari 640x640 ke ukuran gambar asli
            scale_x = image.width / 640
            scale_y = image.height / 640
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            data_tanaman = get_tanaman_by_label(class_name)

            results.append({
                "label": class_name,
                "score": round(float(score), 4),
                "box": [x1, y1, x2, y2],
                "data_tanaman": data_tanaman or "Data not found"
            })

    return JSONResponse(content={"message": "success", "detections": results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5002, reload=False)

