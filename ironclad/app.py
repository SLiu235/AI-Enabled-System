from flask import Flask, request, jsonify
import logging
import os
from PIL import Image
import numpy as np
from datetime import datetime
from modules.retrieval.pipeline import Pipeline
import json

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the pipeline globally
pipeline = Pipeline(pretrained='vggface2', device='cpu')
pipeline.process_gallery('storage/multi_image_gallery', 'vgg')
pipeline.load_embeddings('vgg')
print("Pipeline initialized successfully")

# Store search history
search_history = []

class SearchRecord:
    def __init__(self, probe_image_name, results, timestamp):
        self.probe_image_name = probe_image_name
        self.results = results
        self.timestamp = timestamp

    def to_dict(self):
        return {
            'probe_image_name': self.probe_image_name,
            'results': self.results,
            'timestamp': self.timestamp.isoformat()
        }

@app.route("/identify", methods=['POST'])
def identify():
    """
    Endpoint to process a probe image and return top-k similar images
    
    Parameters:
        - file: The probe image file
        - k: Number of top matches to return (default: 5)
    
    Returns:
        JSON with top-k matches and their similarity scores
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    k = int(request.args.get('k', default=5))
    
    try:
        # Process the probe image
        probe_image = Image.open(file)
        
        # Get the results
        results = pipeline.search_gallery(probe_image, k=k)
        name_results = []
        for result in results:
            name_results.append(result['name'])

        logging.info(f"Results: {name_results}, Type: {type(name_results)}")
        
        # Record the search
        record = SearchRecord(
            probe_image_name=file.filename,
            results=name_results,
            timestamp=datetime.now()
        )
        search_history.append(record)
        
        return jsonify({
            'matches': name_results,
            'timestamp': record.timestamp.isoformat()
        })
        
    except Exception as e:
        logging.error(f"Error in identification: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/add", methods=['POST'])
def add():
    """
    Endpoint to add new images to the gallery
    
    Parameters:
        - file: The image file to add
        - identity: The identity label for the image
    
    Returns:
        JSON confirmation message
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    identity = request.form.get('identity')
    
    if not identity:
        return jsonify({'error': 'No identity provided'}), 400
    
    try:
        # Save the image to the gallery
        save_path = os.path.join('storage/multi_image_gallery', identity)
        os.makedirs(save_path, exist_ok=True)
        
        image_path = os.path.join(save_path, f"{identity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        file.save(image_path)
        
        # Update the pipeline
        pipeline.process_gallery('storage/multi_image_gallery', 'vgg')
        pipeline.load_embeddings('vgg')
        
        return jsonify({
            'message': 'Successfully added to gallery',
            'path': image_path
        })
        
    except Exception as e:
        logging.error(f"Error adding to gallery: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/history", methods=['GET'])
def history():
    """
    Endpoint to retrieve search history
    
    Parameters:
        - limit: Maximum number of records to return (optional)
        - start_date: Filter records from this date (optional, ISO format)
        - end_date: Filter records until this date (optional, ISO format)
    
    Returns:
        JSON list of historical searches
    """
    try:
        limit = request.args.get('limit', type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        filtered_history = search_history
        
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
            filtered_history = [
                record for record in filtered_history 
                if record.timestamp >= start_dt
            ]
            
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
            filtered_history = [
                record for record in filtered_history 
                if record.timestamp <= end_dt
            ]
            
        if limit:
            filtered_history = filtered_history[-limit:]
            
        return jsonify({
            'history': [record.to_dict() for record in filtered_history]
        })
        
    except Exception as e:
        logging.error(f"Error retrieving history: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)