FROM kuralabs/python3-dev:latest

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY BinanceDetectMoonings.py .

CMD [ "python3", "./BinanceDetectMoonings.py" ]