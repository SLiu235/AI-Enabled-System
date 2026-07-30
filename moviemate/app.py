import os
from typing import List, Dict
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel
from surprise import SVD
import uvicorn
import numpy as np
from scipy.stats import ks_2samp

from pipeline import Pipeline
from modules.adaptive.filters.collaborative import CollaborativeFiltering
from modules.adaptive.continuous_learning import ContinuousLearner
from modules.personalization.recommender import Recommender
from modules.personalization.diversifier import Diversifier

# User Management
class User(BaseModel):
    username: str
    user_id: int
    preferences: List[str] = []

class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    password: str
    initial_preferences: List[str] = []

class ModelRetrainingRequest(BaseModel):
    production_rmse: List[float]
    retrain_threshold: float = 0.05

USERS = {}

# Model Configuration
SVD_PARAMS = {
    'n_factors': 200, 
    'n_epochs': 100, 
    'lr_all': 0.01, 
    'reg_all': 0.1
}

# Initialize Recommender System
pipeline = Pipeline()
ratings_df = pipeline.load_data('storage/u.data')
train_df, test_df = pipeline.partition_data(
    ratings_df, 
    partition_type='temporal', 
    save_path='storage', 
    data_name='v1'
)

# Collaborative Filtering Model
model = CollaborativeFiltering(
    algorithm=SVD(**SVD_PARAMS),
    ratings_file='storage/train/v1',
    metadata_file='storage/u.item'
)
model.fit()

recommender = Recommender(model=model)
diversifier = Diversifier(
    metadata=model.items_metadata, 
    lambda_diversity=0.5
)

# Initialize Continuous Learner with baseline RMSE
baseline_rmse = [model.evaluate()]
continuous_learner = ContinuousLearner(baseline_rmse)

# FastAPI Application
app = FastAPI(title="MovieMate Recommender System")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_user(username: str, password: str, initial_preferences: List[str] = []) -> User:
    """Create a new user with a unique ID."""
    if username in USERS:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Find the next available user ID
    user_id = len(USERS) + 1
    
    user = User(
        username=username, 
        user_id=user_id, 
        preferences=initial_preferences
    )
    USERS[username] = {"user": user, "password": password}
    return user

def authenticate_user(username: str, password: str) -> User:
    """Authenticate user credentials."""
    user_data = USERS.get(username)
    if not user_data or user_data['password'] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user_data['user']

@app.post("/users/register")
def register_user(user_data: UserCreate):
    """Register a new user in the system."""
    user = create_user(
        username=user_data.username, 
        password=user_data.password, 
        initial_preferences=user_data.initial_preferences
    )
    return {"message": "User registered successfully", "user_id": user.user_id}

@app.post("/users/login")
def login(user_data: UserLogin):
    """Login and retrieve user information."""
    user = authenticate_user(user_data.username, user_data.password)
    return {"message": "Login successful", "user_id": user.user_id}

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, top_n: int = 10):
    """Generate personalized movie recommendations."""
    try:
        # Get initial rankings
        rankings = recommender.rank_items(user_id, top_n)
        
        # Apply diversity re-ranking
        diverse_rankings = diversifier.rerank(rankings, top_n)
        
        # Convert to movie titles
        movie_recommendations = diverse_rankings.merge(
            model.items_metadata[['item', 'title']], 
            on='item'
        )
        
        return movie_recommendations[['title', 'score']].to_dict(orient='records')
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/detect-drift")
def detect_model_drift(request: ModelRetrainingRequest):
    """
    Detect model performance drift and recommend retraining.
    
    Args:
        request (ModelRetrainingRequest): Contains production RMSE values
    
    Returns:
        Dict with drift detection results
    """
    try:
        # Detect drift
        needs_retraining, p_value = continuous_learner.detect_drift(
            request.production_rmse, 
            return_pvalue=True
        )
        
        return {
            "needs_retraining": needs_retraining, 
            "p_value": p_value,
            "threshold": request.retrain_threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/retrain")
def retrain_model():
    """
    Retrain the recommendation model with the latest data.
    
    Returns:
        Dict with retraining status and new baseline RMSE
    """
    try:
        # Load latest data
        ratings_df = pipeline.load_data('storage/u.data')
        train_df, _ = pipeline.partition_data(
            ratings_df, 
            partition_type='temporal', 
            save_path='storage', 
            data_name='latest'
        )
        
        # Retrain model
        global model, recommender, continuous_learner
        model = CollaborativeFiltering(
            algorithm=SVD(**SVD_PARAMS),
            ratings_file='storage/train/latest',
            metadata_file='storage/u.item'
        )
        model.fit()
        
        # Update recommender and diversifier
        recommender = Recommender(model=model)
        diversifier = Diversifier(
            metadata=model.items_metadata, 
            lambda_diversity=0.5
        )
        
        # Update baseline RMSE for drift detection
        baseline_rmse = [model.evaluate()]
        continuous_learner = ContinuousLearner(baseline_rmse)
        
        return {
            "message": "Model retrained successfully", 
            "new_baseline_rmse": baseline_rmse[0]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)