FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git wget libgl1 && \
    rm -rf /var/lib/apt/lists/*

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 只复制必要文件（避免垃圾）
COPY app.py .
COPY faiss_index ./faiss_index

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]