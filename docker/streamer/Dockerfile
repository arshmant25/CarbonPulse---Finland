FROM python:3.13.14-slim
WORKDIR /app
RUN pip install --no-cache-dir requests==2.34.2 pandas==3.0.3 schedule==1.2.2
COPY streamer.py .
CMD ["python", "streamer.py"]
