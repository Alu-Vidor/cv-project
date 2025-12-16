import os
from typing import Optional

import requests
import streamlit as st

st.set_page_config(page_title="AI Calorie Counter", page_icon="🍎", layout="wide")

st.title("AI Calorie Counter 🍎")
st.write(
    "Загрузите фотографию блюда или сделайте снимок с веб-камеры, чтобы узнать его калорийность."
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"


def _choose_image(upload: Optional[st.uploaded_file_manager.UploadedFile],
                  camera: Optional[st.uploaded_file_manager.UploadedFile]):
    if upload is not None:
        return upload
    if camera is not None:
        return camera
    return None


def _format_calories(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if numeric.is_integer():
        return f"{int(numeric)} ккал"
    return f"{numeric:.1f} ккал"


uploaded_file = st.file_uploader(
    "Загрузите фото блюда",
    type=["jpg", "jpeg", "png"],
    help="Поддерживаются форматы JPG и PNG",
)
camera_photo = st.camera_input("Или сделайте фото с веб-камеры")

image_file = _choose_image(uploaded_file, camera_photo)

if image_file is not None:
    st.subheader("Ваше изображение")
    st.image(image_file, use_column_width=True)

    with st.spinner("Определяем блюдо и калории..."):
        try:
            files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
            response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
            response.raise_for_status()
            prediction = response.json()
        except requests.RequestException:
            st.error("Не удалось получить ответ от сервера. Попробуйте позже.")
            st.stop()
        except ValueError:
            st.error("Получен некорректный ответ от сервера.")
            st.stop()

    food_name = prediction.get("food_name") or "Неизвестное блюдо"
    calories_display = _format_calories(prediction.get("calories"))

    st.markdown(f"## {food_name}")
    st.metric(label="Калорийность", value=calories_display)
else:
    st.info("Загрузите изображение или сделайте снимок, чтобы начать.")
