# MovieMate: A Movie Recommendation System

## Overview
MovieMate is an advanced movie recommendation system that uses collaborative filtering, content-based techniques, and machine learning to provide personalized and diverse movie recommendations.

## Installation

### Local Setup
1. Clone the repository:
```bash
git clone https://github.com/creating-ai-enabled-systems-fall-2024/liu-sichao/moviemate
cd moviemate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

### Docker Setup
1. Build the Docker image:
```bash
docker build -t moviemate .
```

2. Run the container:
```bash
docker run -d \
  -p 8000:8000 \
  moviemate
```

## API Endpoints

### User Registration
```bash
curl -X POST http://localhost:8000/users/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "123456",
    "password": "secure_password",
    "initial_preferences": ["Action", "Sci-Fi"]
  }'
```

### User Login
```bash
curl -X POST http://localhost:8000/users/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "123456",
    "password": "secure_password"
  }'
```

### Get Recommendations
```bash
curl -X GET http://localhost:8000/recommendations/1?top_n=10
```

### Drift detection:
```bash
curl -X POST http://localhost:8000/model/detect-drift \
  -H "Content-Type: application/json" \
  -d '{
    "production_rmse": [0.85, 0.90, 0.88],
    "retrain_threshold": 0.05
  }'
```

### Model retraining:
```bash
curl -X POST http://localhost:8000/model/retrain
```


## Configuration Options
* `top_n`: Number of recommendations to retrieve (default: 10)
* `lambda_diversity`: Balance between relevance and diversity (default: 0.5)
