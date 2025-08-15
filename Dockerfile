FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 設定環境變數，避免互動式提示
ENV TZ=Asia/Taipei
ENV DEBIAN_FRONTEND=noninteractive

# 安裝 Python, pip, 以及音訊處理必要的 FFmpeg
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    software-properties-common && \
    add-apt-repository ppa:ubuntuhandbook1/ffmpeg7 && \
    apt-get update && apt-get install -y ffmpeg

# 建立一個符號連結，讓 'python' 指向 'python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# 設定工作目錄
WORKDIR /app

# 複製依賴清單並安裝
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
