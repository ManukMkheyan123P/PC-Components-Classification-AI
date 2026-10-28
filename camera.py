
import cv2
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque

# =====================================
# 1. ЗАГРУЗКА МОДЕЛИ И КЛАССОВ
# =====================================

BASE_DIR = Path(__file__).resolve().parent

model = tf.keras.models.load_model(
    BASE_DIR / "electronics_ai_v2.keras",
    compile=False
)

with open(BASE_DIR / "classes.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        "Количество классов не совпадает с выходом модели!"
    )

print("Electronics AI V2 загружена!")
print("Классы:", class_names)

# =====================================
# 2. ЗАПУСК КАМЕРЫ
# =====================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть веб-камеру")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Усредняем предсказания последних 5 кадров,
# чтобы название устройства меньше прыгало
prediction_history = deque(maxlen=5)

print("Камера запущена! Нажми Q для выхода.")

# =====================================
# 3. РАСПОЗНАВАНИЕ В РЕАЛЬНОМ ВРЕМЕНИ
# =====================================

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Не удалось получить кадр")
            break

        h, w = frame.shape[:2]

        # Центральная квадратная область
        side = int(min(h, w) * 0.75)

        x1 = (w - side) // 2
        y1 = (h - side) // 2
        x2 = x1 + side
        y2 = y1 + side

        roi = frame[y1:y2, x1:x2]

        # OpenCV использует BGR, модель ожидает RGB
        image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SIZE)

        # В модели уже есть Rescaling в [-1, 1]
        # Поэтому НЕ делим изображение на 255
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        # Предсказание
        predictions = model(
            image,
            training=False
        ).numpy()[0]

        prediction_history.append(predictions)

        # Усредняем вероятности
        avg_predictions = np.mean(
            prediction_history,
            axis=0
        )

        predicted_index = int(np.argmax(avg_predictions))
        confidence = float(avg_predictions[predicted_index])

        predicted_class = class_names[predicted_index]

        if confidence >= CONFIDENCE_THRESHOLD:
            label = f"{predicted_class.upper()} - {confidence * 100:.1f}%"
            color = (0, 255, 0)
        else:
            label = f"Uncertain: {predicted_class} - {confidence * 100:.1f}%"
            color = (0, 165, 255)

        # Рамка вокруг области распознавания
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Чёрный фон для текста
        cv2.rectangle(
            frame,
            (10, 10),
            (min(w - 10, 650), 60),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (20, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Electronics AI - Live Recognition", frame)

        # Q - выход
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
