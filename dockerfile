FROM tensorflow/tensorflow:2.10.0-gpu
WORKDIR /app
COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir /data
