<div align="center">

# AI-Enabled Systems Portfolio

### A Collection of Production-Style Machine Learning Systems

</div>

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Repository Structure](#2-repository-structure)
3. [System Overview](#3-system-overview)
4. [System Descriptions](#4-system-descriptions)
   - [4.1 IronClad — Visual Search & Facial Recognition](#41-ironclad--visual-search--facial-recognition)
   - [4.2 MovieMate — Movie Recommendation System](#42-moviemate--movie-recommendation-system)
   - [4.3 SecureBank — Fraud Detection System](#43-securebank--fraud-detection-system)
   - [4.4 TechTrack — Object Detection System](#44-techtrack--object-detection-system)
   - [4.5 Textwave — Document QA & Semantic Search](#45-textwave--document-qa--semantic-search)
5. [Port & Environment Reference](#5-port--environment-reference)
6. [Requirements](#6-requirements)
7. [Notes on Usage](#7-notes-on-usage)

---

## 1. Introduction

This repository consolidates five independently developed, containerized machine learning systems, each completed as part of the *Creating AI-Enabled Systems* coursework. While each system addresses a distinct problem domain — facial recognition, recommendation, fraud detection, object detection, and document question-answering — all five share a common engineering philosophy:

- **Modular architecture**, separating data extraction, processing, and retrieval/inference logic
- **Containerization** via Docker for reproducible deployment
- **RESTful APIs** exposing core functionality for integration and testing
- **System documentation** (`README.md` and, where applicable, `System_Report.md`) accompanying each project

Each system resides in its own subdirectory and can be built, run, and evaluated independently of the others. This document serves as the top-level entry point to the portfolio, summarizing each system's purpose, architecture, and usage.

---

## 2. Repository Structure

```bash
.
├── ironclad/       # System I   — Visual search & facial recognition
├── moviemate/      # System II  — Movie recommendation system
├── securebank/     # System III — Fraud detection system
├── techtrack/      # System IV  — Object detection system
└── textwave/       # System V   — Document QA / semantic search system
```

Each subdirectory contains, at minimum:

- `app.py` — application entry point
- `Dockerfile` — container build specification
- `requirements.txt` — Python dependencies
- `README.md` — system-specific documentation
- `System_Report.md` — design rationale and evaluation, where applicable

---

## 3. System Overview

| # | System | Domain | Core Technique | Interface |
|---|--------|--------|-----------------|-----------|
| I | IronClad | Computer Vision | Facial embedding & similarity search | REST API |
| II | MovieMate | Recommender Systems | Collaborative filtering + content-based ML | REST API |
| III | SecureBank | Anomaly Detection | Supervised fraud classification | REST API |
| IV | TechTrack | Computer Vision | Object detection, hard-negative mining | CLI + UDP stream |
| V | Textwave | NLP / Information Retrieval | Semantic search (FAISS) + LLM generation | REST API |

---

## 4. System Descriptions

### 4.1 IronClad — Visual Search & Facial Recognition

**Purpose:** IronClad is a visual search and facial recognition system that allows users to identify individuals from probe images, register new identities to a gallery, and retrieve search history.

**Architecture**
```bash
ironclad/
├── modules/
│   ├── extraction/     # embedding.py, preprocess.py
│   └── retrieval/      # indexing.py, pipeline.py, search.py
├── storage/
├── app.py
├── Dockerfile
├── requirements.txt
└── System_Report.md
```

**Deployment**
```bash
git clone https://github.com/creating-ai-enabled-systems-fall-2024/liu-sichao/ironclad.git
cd ironclad
docker build -t face-recognition-app .
docker run -d --name face-recognition-service -p 5001:5001 \
  -v $(pwd)/storage:/app/storage face-recognition-app
```

**API Reference**

| Function | Endpoint | Example |
|---|---|---|
| Identify a face | `POST /identify` | `curl -X POST -F "file=@probe_image.jpg" http://localhost:5001/identify?k=5` |
| Add an identity | `POST /add` | `curl -X POST -F "file=@new_image.jpg" -F "identity=person_name" http://localhost:5001/add` |
| Retrieve history | `GET /history` | `curl "http://localhost:5001/history?limit=10"` |

---

### 4.2 MovieMate — Movie Recommendation System

**Purpose:** MovieMate is a hybrid recommendation engine that combines collaborative filtering and content-based methods to deliver personalized, diverse movie recommendations, with monitoring for model drift and automated retraining.

**Deployment — Local**
```bash
git clone https://github.com/creating-ai-enabled-systems-fall-2024/liu-sichao/moviemate
cd moviemate
pip install -r requirements.txt
python app.py
```

**Deployment — Docker**
```bash
docker build -t moviemate .
docker run -d -p 8000:8000 moviemate
```

**API Reference**

| Function | Endpoint | Parameters |
|---|---|---|
| Register user | `POST /users/register` | `username`, `password`, `initial_preferences` |
| Login | `POST /users/login` | `username`, `password` |
| Get recommendations | `GET /recommendations/{user_id}` | `top_n` |
| Detect model drift | `POST /model/detect-drift` | `production_rmse`, `retrain_threshold` |
| Retrain model | `POST /model/retrain` | — |

**Configuration Parameters**

| Parameter | Description | Default |
|---|---|---|
| `top_n` | Number of recommendations returned | 10 |
| `lambda_diversity` | Balance between relevance and diversity | 0.5 |

---

### 4.3 SecureBank — Fraud Detection System

**Purpose:** SecureBank applies machine learning to classify financial transactions as fraudulent or legitimate, with an end-to-end pipeline covering dataset generation, model training, model selection, prediction, and performance auditing.

**Deployment**
```bash
docker build -t securebank .
docker run -p 5001:5001 securebank
```
The service becomes available at `http://localhost:5001`.

**API Reference**

| Endpoint | Method | Description | Required Parameters |
|---|---|---|---|
| `/help` | GET | Lists all available commands | — |
| `/predict/` | POST | Classifies a transaction as fraudulent or legitimate | `trans_date_trans_time`, `cc_num`, `unix_time`, `merchant`, `category`, `amt`, `merch_lat`, `merch_long` |
| `/generate_dataset/` | POST | Generates a new dataset version | `version`, `num_customers`, `num_transactions`, `fraud_ratio` |
| `/train_model/` | POST | Trains a model on a specified dataset | `model_name`, `dataset_version` |
| `/select_model/` | POST | Selects a trained model for inference | `model_name` |
| `/history` | GET | Returns prediction history | — |
| `/audit_performance/` | POST | Reports false positive / false negative rates | `dataset_version` |

**Recommended Workflow**

1. Generate a dataset via `/generate_dataset/`
2. Train a model via `/train_model/`
3. Select the trained model via `/select_model/`
4. Submit transactions for classification via `/predict/`
5. Evaluate model performance via `/audit_performance/`

---

### 4.4 TechTrack — Object Detection System

**Purpose:** TechTrack implements an end-to-end object detection pipeline supporting three operational modes: batch inference, dataset rectification (hard-negative mining and augmentation), and real-time UDP streaming.

**Architecture**
```bash
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
└── System_Report.md
```

**Deployment**
```bash
git clone https://github.com/creating-ai-enabled-systems-fall-2024/liu-sichao/techtrack.git
cd techtrack
docker build -t techtrack .
```

**Operational Modes**

| Mode | Purpose | Command |
|---|---|---|
| Inference | Run object detection on a video file | `python app.py inference --input /app/input/video.mp4 --output /app/output` |
| Rectification | Perform hard-negative mining and data augmentation | `python app.py rectification --input /app/input --output /app/output --top_n 10` |
| Streaming | Launch a real-time UDP streaming server | `python app.py stream --host 127.0.0.1 --port 23000` |

Inference and rectification modes require volume mounts:
```bash
docker run -v /path/to/input:/app/input -v /path/to/output:/app/output techtrack
```

Streaming mode requires the UDP port to be exposed:
```bash
docker run -p 23000:23000/udp techtrack python app.py stream --host 127.0.0.1 --port 23000
```

---

### 4.5 Textwave — Document QA & Semantic Search

**Purpose:** Textwave is a FastAPI-based question-answering system that ingests documents, performs semantic search using sentence-transformer embeddings and FAISS indexing, reranks results with a cross-encoder, and generates natural-language answers using Mistral AI.

**Architecture**
```bash
textwave/
├── app.py
├── Dockerfile
├── requirements.txt
├── storage/
│   ├── corpus/
│   └── index/
└── modules/
    ├── extraction/
    ├── retrieval/
    └── generator/
```

**Core Capabilities**
- Document upload and processing
- Configurable text chunking strategies
- Semantic search via FAISS indexing
- Cross-encoder reranking
- Answer generation via Mistral AI

**Deployment — Local**
```bash
git clone https://github.com/creating-ai-enabled-systems-fall-2024/liu-sichao/textwave.git
cd textwave
pip install -r requirements.txt
export MISTRAL_API_KEY="your-api-key-here"
```

**Deployment — Docker**
```bash
docker build -t textwave .
docker run -d -p 8000:8000 \
  -v $(pwd)/storage:/app/storage \
  -e MISTRAL_API_KEY="your-api-key-here" \
  textwave
```

**API Reference**

| Function | Endpoint | Example |
|---|---|---|
| Upload document | `POST /upload_document` | `curl -X POST "http://localhost:8000/upload_document" -F "file=@document.pdf"` |
| Process corpus | `POST /process_corpus` | `curl -X POST "http://localhost:8000/process_corpus" -d "chunking_strategy=sentence&overlap_size=2"` |
| Ask a question | `POST /ask` | `curl -X POST "http://localhost:8000/ask" -d "question=Your question here?&k=5&rerank=true"` |

**Interactive Documentation**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

**Configuration Parameters**

| Category | Parameter | Description | Default |
|---|---|---|---|
| Corpus Processing | `chunking_strategy` | `sentence` or `fixed_length` | — |
| Corpus Processing | `fixed_length` | Token count for fixed-length chunking | — |
| Corpus Processing | `overlap_size` | Overlap between adjacent chunks | — |
| Question Answering | `k` | Number of documents retrieved | 5 |
| Question Answering | `rerank` | Enable/disable cross-encoder reranking | true |

---

## 5. Port & Environment Reference

| System | Default Port | Protocol | Notes |
|---|---|---|---|
| IronClad | 5001 | HTTP | Conflicts with SecureBank on default port |
| MovieMate | 8000 | HTTP | Conflicts with Textwave on default port |
| SecureBank | 5001 | HTTP | Conflicts with IronClad on default port |
| TechTrack | 23000 | UDP | Streaming mode only |
| Textwave | 8000 | HTTP | Conflicts with MovieMate on default port; requires `MISTRAL_API_KEY` |

> **Important:** IronClad and SecureBank share port 5001 by default; MovieMate and Textwave share port 8000 by default. When running more than one system concurrently, remap host ports using `-p <host_port>:<container_port>` to avoid conflicts.

---

## 6. Requirements

- **Docker** — required for containerized deployment of all five systems
- **Python 3.x** and **pip** — required only for local (non-Docker) execution
- **Mistral AI API key** — required for Textwave's answer-generation module

---

## 7. Notes on Usage

This repository is organized to allow each system to be evaluated independently. Reviewers and graders should navigate to the relevant subdirectory for system-specific implementation details, and consult each system's `System_Report.md` (where present) for design decisions, evaluation methodology, and performance results.

<div align="center">

---

</div>
