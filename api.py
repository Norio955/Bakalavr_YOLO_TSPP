from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Створюємо додаток (FastAPI автоматично згенерує з цього Swagger)
app = FastAPI(
    title="Road Signs Detection API",
    description="API для автоматичного розпізнавання дорожніх знаків за допомогою комп'ютерного зору (YOLOv11).",
    version="1.0.0",
)

# 2. Завантажуємо твою модель YOLOv5
model = YOLO("best.pt")


# 3. Створюємо ендпоінт
@app.post("/detect", summary="Розпізнати знаки на завантаженому фото")
async def detect_signs(file: UploadFile = File(...)):
    """
    Завантажте зображення (JPG/PNG), і нейромережа поверне список знайдених знаків.

    Цей багаторядковий коментар автоматично стане частиною живої документації Swagger!
    """
    # Читаємо завантажену картинку
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Передаємо в YOLO
    results = model(img)
    detections = []

    # Витягуємо результати
    for box in results[0].boxes:
        detections.append({"class_name": results[0].names[int(box.cls)], "confidence": round(float(box.conf) * 100, 2)})

    return {"status": "success", "detections": detections}
