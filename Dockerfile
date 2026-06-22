FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# Copy backend code
COPY backend/ .

# Copy frontend static files (托管聊天页面)
COPY frontend/ ./frontend/


EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
