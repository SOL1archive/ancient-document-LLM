FROM python:latest

COPY requirements.txt .
COPY run.sh .
COPY crawler.py .
COPY sillok_crawler.py .
COPY sillok-crawler-config.yaml .

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    unzip \
    fonts-liberation \
    libappindicator3-1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxrandr2 \
    xdg-utils \
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt update && apt install -y ./google-chrome-stable_current_amd64.deb
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
RUN CHROME_DRIVER_VERSION=$(wget -qO- https://chromedriver.storage.googleapis.com/LATEST_RELEASE) && \
    wget -q "https://chromedriver.storage.googleapis.com/${CHROME_DRIVER_VERSION}/chromedriver_linux64.zip" -O /tmp/chromedriver.zip && \
    unzip /tmp/chromedriver.zip -d /usr/bin && \
    rm /tmp/chromedriver.zip
RUN pip install -r requirements.txt
RUN chmod +x run.sh

ENTRYPOINT [ "./run.sh" ]
