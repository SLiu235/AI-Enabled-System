# TechTrack Object Detection System

This project implements an object detection system with inference, rectification, and streaming capabilities.

## Project Structure

```
techtrack/
├── inference/
│   ├── nms.py
│   ├── object_detection.py
│   ├── preprocessing.py
│   └── udp_stream.py
├── rectification/
│   ├── hard_negative_mining.py
│   └── augmentation.py
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
└── System_Report.md
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/creating-ai-enabled-systems-fall-2024/liu-sichao/techtrack.git
   cd techtrack
   ```

2. Build the Docker image:
   ```
   docker build -t techtrack .
   ```

## Running the Project

The project can be run in three modes: inference, rectification, and streaming.

### Inference Mode

To run object detection on a video:

```
docker run -v /path/to/input:/app/input -v /path/to/output:/app/output techtrack
```
```
 python app.py inference --input /app/input/video.mp4 --output /app/output
```

### Rectification Mode

To perform hard negative mining and augmentation:

```
docker run -v /path/to/input:/app/input -v /path/to/output:/app/output techtrack 
```
```
python app.py rectification --input /app/input --output /app/output --top_n 10
```

### Streaming Mode

To start the UDP streaming server:

```
docker run -p 23000:23000/udp techtrack python app.py stream --host 127.0.0.1 --port 23000
```
```
python app.py stream --host 127.0.0.1 --port 23000
```
