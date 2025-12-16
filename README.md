# CV Project Containers

Этот проект содержит FastAPI backend и Streamlit frontend для подсчета калорий по фотографии блюда. Оба сервиса упакованы в Docker и поднимаются одной командой.

## Быстрый старт
1. Убедитесь, что в корне репозитория находится папка `models/` с файлами модели (`food_classifier.pth`, `classes.json` и др.), чтобы backend мог загрузить веса.
2. Выполните команду:
   ```bash
   docker-compose up --build
   ```
3. Откройте браузер на http://localhost:8501, frontend автоматически обращается к backend по адресу `http://backend:8000` внутри общей сети.

Контейнеры используют общую сеть Docker Compose, backend слушает порт `8000`, frontend — `8501`.
