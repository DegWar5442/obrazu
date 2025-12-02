# Використовуємо легку версію Python
FROM python:3.9-slim

# Встановлюємо робочу папку всередині контейнера
WORKDIR /app

# Копіюємо файл зі списком бібліотек
COPY requirements.txt .

# Встановлюємо бібліотеки
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо ваш скрипт 
COPY lab1_unsupervised_learning.py .

# Ця змінна потрібна, щоб matplotlib не намагався відкрити вікно (бо в Docker немає екрана)
ENV MPLBACKEND=Agg

# Запускаємо скрипт
CMD ["python", "lab1_unsupervised_learning.py"]