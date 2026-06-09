import os
import time
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from query_faiss import perform_search, EmbeddingModels

app = Flask(__name__, template_folder='templates')

# Pre-load models to avoid overhead on every request
print("Loading embedding models...")
MODELS = EmbeddingModels()
print("All models loaded and ready.")

# Base directory for videos
VIDEO_BASE_DIRS = {
    'train': 'videos_train',
    'val': 'videos_val',
    'test': 'videos_test',
    'seen': 'videos_train',
    'unseen': 'videos_test' # Fallback, though metadata split is preferred
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    split = data.get('split', 'test') # Default to test
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        results = perform_search(
            query, 
            split=split, 
            models=MODELS
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    feedback_file = 'artifacts/human_feedback.json'
    
    try:
        os.makedirs('artifacts', exist_ok=True)
        
        feedback_entry = {
            "timestamp_logged": time.strftime("%Y-%m-%d %H:%M:%S"),
            **data
        }
        
        feedbacks = []
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                try:
                    content = f.read()
                    if content:
                        feedbacks = json.loads(content)
                        if not isinstance(feedbacks, list):
                            feedbacks = [feedbacks] # Wrap if it's not a list
                except json.JSONDecodeError:
                    # If file is corrupted, we start with an empty list to avoid crashing
                    feedbacks = []
        
        feedbacks.append(feedback_entry)
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedbacks, f, indent=2)
            
        return jsonify({"status": "success", "message": "Feedback saved"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/video/<split>/<filename>')
def serve_video(split, filename):
    # If the exact split directory is known, use it
    video_dir = VIDEO_BASE_DIRS.get(split)
    if video_dir and os.path.exists(os.path.join(video_dir, filename)):
        return send_from_directory(video_dir, filename)
    
    # Otherwise, search in all possible directories
    for directory in VIDEO_BASE_DIRS.values():
        if os.path.exists(os.path.join(directory, filename)):
            return send_from_directory(directory, filename)
            
    return "Video not found", 404

if __name__ == '__main__':
    # Running on 0.0.0.0 to allow access from outside the container/cluster if needed
    app.run(host='0.0.0.0', port=5000, debug=True)
