FROM python:3.11-slim

ENV MPLBACKEND=Agg

# System deps:
# - wget/gnupg/curl: for installing Chrome
# - tesseract-ocr: for pytesseract
# - libgomp1: required at runtime by lightgbm and xgboost
# - libglib2.0-0: required by opencv-python-headless
# - the rest: Chrome runtime deps
RUN apt-get update && apt-get install -y \
    wget gnupg unzip curl tesseract-ocr \
    libgomp1 libglib2.0-0 \
    fonts-liberation libnss3 libxss1 \
    libasound2 libatk-bridge2.0-0 libgtk-3-0 libappindicator3-1 \
    && wget -q -O /usr/share/keyrings/google-chrome.asc https://dl.google.com/linux/linux_signing_key.pub \
    && echo "deb [signed-by=/usr/share/keyrings/google-chrome.asc] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8501
EXPOSE 8501

CMD streamlit run everything.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
