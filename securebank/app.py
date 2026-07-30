from flask import Flask, request, jsonify, Response
from modules.pipeline import Pipeline
from modules.data_generator import DataGenerator
from modules.model_trainer import ModelTrainer
from modules.performance_auditor import PerformanceAuditor
import time

app = Flask(__name__)
pipeline = Pipeline()
data_generator = DataGenerator()
model_trainer = ModelTrainer()
performance_auditor = PerformanceAuditor()

def generate_prompt(message):
    yield f"{message}\n\n"
    time.sleep(0.5) 

@app.route('/predict/', methods=['POST'])
def predict():
    data = request.json
    required_keys = [
        'trans_date_trans_time', 'cc_num', 'unix_time', 'merchant',
        'category', 'amt', 'merch_lat', 'merch_long'
    ]
    
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Missing required fields"}), 400
    try:
        def generate():
            yield from generate_prompt("Processing transaction...")
            prediction = pipeline.predict(data)
            yield from generate_prompt(f"Prediction: {'fraud' if prediction else 'legitimate'}")
        
        return Response(generate(), content_type='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/generate_dataset/', methods=['POST'])
def generate_dataset():
    version = request.json.get('version')
    if not version:
        return jsonify({"error": "Missing version for dataset"}), 400
    try:
        def generate():
            yield from generate_prompt(f"Generating dataset version {version}...")
            data_generator.generate_and_save_data(version=version)
            yield from generate_prompt(f"Dataset version {version} generated successfully")
        
        return Response(generate(), content_type='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train_model/', methods=['POST'])
def train_model():
    model_name = request.json.get('model_name')
    dataset_version = request.json.get('dataset_version', None)

    if not model_name:
        return jsonify({"error": "Missing model name"}), 400
    try:
        def generate():
            yield from generate_prompt(f"Training model {model_name}...")
            model_trainer.train(model_name, dataset_version)
            yield from generate_prompt(f"Model {model_name} trained successfully")
        
        return Response(generate(), content_type='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/select_model/', methods=['POST'])
def select_model():
    model_name = request.json.get('model_name')
    try:
        def generate():
            yield from generate_prompt(f"Selecting model {model_name}...")
            pipeline.select_model(model_name)
            yield from generate_prompt(f"Model {model_name} selected successfully")
        
        return Response(generate(), content_type='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(pipeline.get_history())

@app.route('/audit_performance/', methods=['POST'])
def audit_performance():
    dataset_version = request.json.get('dataset_version', None)

    try:
        def generate():
            yield from generate_prompt("Auditing model performance...")
            performance_metrics = performance_auditor.audit(pipeline, dataset_version)
            yield from generate_prompt("Performance audit completed")
            yield from generate_prompt(f"Performance metrics: {performance_metrics}")
        
        return Response(generate(), content_type='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/help', methods=['GET'])
def show_help():
    commands = """
    Welcome to the Fraud Detection System. Here are the available commands:

    1. /generate_dataset/ (POST)
       - Description: Generate a new dataset version.
       - Required JSON Parameters:
           - version: str (Dataset version name)
           - num_customers: int (Optional, default 1000)
           - num_transactions: int (Optional, default 10000)
           - fraud_ratio: float (Optional, default 0.005)
       - Example: 
           {
               "version": "v1.0",
               "num_customers": 1000,
               "num_transactions": 10000,
               "fraud_ratio": 0.01
           }

    2. /train_model/ (POST)
       - Description: Train a model with the given dataset version.
       - Required JSON Parameters:
           - model_name: str (One of: "logistic_regression", "rvm", "random_forest")
           - dataset_version: str (Version of the dataset to use)
       - Example: 
           {
               "model_name": "random_forest",
               "dataset_version": "v1.0"
           }

    3. /select_model/ (POST)
       - Description: Select a pre-trained model for prediction.
       - Required JSON Parameters:
           - model_name: str (Model name to select)
       - Example: 
           {
               "model_name": "random_forest"
           }

    4. /predict/ (POST)
       - Description: Predict if a transaction is fraudulent.
       - Required JSON Parameters:
           - Transaction data with the following fields:
               - trans_date_trans_time, cc_num, unix_time, merchant,
                 category, amt, merch_lat, merch_long
       - Example: 
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

    5. /history (GET)
       - Description: Get the prediction history.
       - No parameters required.

    6. /audit_performance/ (POST)
       - Description: Audit the performance of the selected model on a dataset.
       - Required JSON Parameters:
           - dataset_version: str (Version of the dataset to audit)
       - Example: 
           {
               "dataset_version": "v1.0"
           }
    """
    return commands

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)