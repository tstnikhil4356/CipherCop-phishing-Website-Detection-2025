FROM python:3.11-slim

# System deps: Chrome (for Selenium), chromedriver, Tesseract OCR
RUN apt-get update && apt-get install -y \
    wget gnupg unzip curl tesseract-ocr \
    fonts-liberation libnss3 libgconf-2-4 libxss1 libappindicator3-1 \
    libasound2 libatk-bridge2.0-0 libgtk-3-0 \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8501
EXPOSE 8501

CMD streamlit run everything.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
