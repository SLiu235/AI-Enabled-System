# IronClad Visual Search System 

IronClad is a visual search and facial recognition system. It enables users to identify and add images to a gallery and view search history.

## Project Structure

```bash
ironclad/
├── modules/
│   ├── extraction/
|   |       ├── embedding.py
|   |       └── preprocess.py
│   └── retrieval/
|           ├── indexing.py
|           ├── pipeline.py
|           └── search.py
├── storage/
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
└── System_Report.md
```

## Setup

### 1. Clone the repository:

```bash
git clone  https://github.com/creating-ai-enabled-systems-fall-2024/liu-sichao/ironclad.git
cd ironclad
```

### 2. Build the Docker Image:

```bash
docker build -t face-recognition-app .
```

### 3. Run the Docker Container

```bash
docker run -d \
  --name face-recognition-service \
  -p 5001:5001 \
  -v $(pwd)/storage:/app/storage \
  face-recognition-app
```

## Running the Project

The project can be run in three modes: identify an image, add an image, and check the history.

### Identify a Face

```bash
curl -X POST \
  -F "file=@path/to/probe_image.jpg" \
  http://localhost:5001/identify?k=5
```

### Add a New Identity to Gallery

```bash
curl -X POST \
  -F "file=@path/to/new_image.jpg" \
  -F "identity=person_name" \
  http://localhost:5001/add
```

### Get Search History

```bash
curl "http://localhost:5001/history?limit=10"
```





