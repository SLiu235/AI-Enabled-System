# SecureBank Fraud Detection System

This system is designed to detect fraudulent transactions using machine learning techniques.

## Requirements

- Docker

## How to Run

1. Clone this repository
2. Navigate to the project directory
3. Build the Docker image:
   ```
   docker build -t securebank .
   ```
4. Run the Docker container:
   ```
   docker run -p 5001:5001 securebank
   ```

The system will now be running on `http://localhost:5001`.


## API Endpoints and Usage

### 1. `/help` (GET)

Displays all available commands and how to use them.

### 2. `/predict/` (POST)

Predicts if a given transaction is fraudulent or legitimate.

- **Required JSON Parameters:**
    ```json
    {
        "trans_date_trans_time": "2024-09-21 12:34:56",
        "cc_num": 1234567890123456,
        "unix_time": 1622547800,
        "merchant": "Online Store",
        "category": "Shopping",
        "amt": 100.50,
        "merch_lat": 40.7128,
        "merch_long": -74.0060
    }
    ```

- **Response:**
    - Returns whether the transaction is predicted as "fraudulent" or "legitimate".

### 3. `/generate_dataset/` (POST)

Generates a new dataset version with the specified parameters.

- **Required JSON Parameters:**
    ```json
    {
        "version": "v1.0",
        "num_customers": 1000,
        "num_transactions": 10000,
        "fraud_ratio": 0.01
    }
    ```

- **Response:**
    - Confirms the successful generation of the specified dataset version.

### 4. `/train_model/` (POST)

Trains a specified model using a selected dataset version.

- **Required JSON Parameters:**
    ```json
    {
        "model_name": "random_forest",
        "dataset_version": "v1.0"
    }
    ```

- **Response:**
    - Confirms that the model has been trained successfully.

### 5. `/select_model/` (POST)

Selects a pre-trained model for future predictions.

- **Required JSON Parameters:**
    ```json
    {
        "model_name": "random_forest"
    }
    ```

- **Response:**
    - Confirms that the specified model has been selected successfully.

### 6. `/history` (GET)

Fetches the prediction history, showing previously predicted transactions and their results.

- **Response:**
    - Returns the prediction history as a list of transactions with outcomes.

### 7. `/audit_performance/` (POST)

Audits the performance of the currently selected model using the specified dataset version.

- **Required JSON Parameters:**
    ```json
    {
        "dataset_version": "v1.0"
    }
    ```

- **Response:**
    - Returns performance metrics false positive rate and false negative rate for the model on the specified dataset.

## Usage Tips

- Start by generating a dataset using the `/generate_dataset/` endpoint.
- Train a model using the `/train_model/` endpoint, specifying the model type and dataset version.
- Use the `/select_model/` endpoint to choose a trained model for making predictions.
- Predict transactions with the `/predict/` endpoint to identify potential fraud.
- Audit model performance using the `/audit_performance/` endpoint to evaluate how well your model performs on different datasets.

