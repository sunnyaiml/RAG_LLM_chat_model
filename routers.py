from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pickle
from optimize import load_document, create_vector_store, create_custom_qa_chain
from optimize import(
    save_qa_chain_to_pickle,
    load_qa_chain_from_pickle,
    load_document,
    split_text,
    create_vector_store,
    create_custom_qa_chain

)
# Initialize Flask app
app = Flask(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Base directory of the app
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')  # Directory to store uploaded files
PICKLE_FOLDER = os.path.join(BASE_DIR, 'pickle')  # Directory to store pickle files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure folder exists
os.makedirs(PICKLE_FOLDER, exist_ok=True)  # Ensure folder exists

# Global variables
vector_store = None
qa_chain = None


# Route to serve the HTML page
@app.route('/')
def home():
    """
    Renders the main HTML page for the frontend.
    Make sure the HTML file is in a templates/ directory relative to this script.
    """
    return render_template('frontend\templates\index.html')  # Adjust as needed


# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint to handle file uploads.
    Stores uploaded files in the UPLOAD_FOLDER and validates file types.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Allow only PDF and TXT files
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        return jsonify({"error": "Only PDF and TXT files are allowed"}), 400

    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({"message": f"File {file.filename} uploaded successfully"}), 200


# Route to train the model
@app.route('/train', methods=['POST'])
def train_model():
    """
    Trains the model using the uploaded file.
    Processes the file into chunks, creates vector store, and saves the QA chain.
    """
    global vector_store, qa_chain

    # Ensure there is an uploaded file
    files = os.listdir(UPLOAD_FOLDER)
    if len(files) == 0:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, files[0])  # Pick the first uploaded file
    try:
        # Load and process document
        documents = load_document(file_path)
        chunks = create_vector_store(documents)

        # Create vector store
        vector_store = create_vector_store(chunks)
        if vector_store is None:
            return jsonify({"error": "Failed to create vector store"}), 500

        # Create QA chain
        qa_chain = create_custom_qa_chain(vector_store)

        # Save the QA chain to a pickle file
        pickle_path = os.path.join(PICKLE_FOLDER, 'qa_chain.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(qa_chain, f)

        return jsonify({"message": "Model trained successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route to serve static files (if required)
@app.route('/static/<path:filename>')
def serve_static(filename):
    """
    Serves static files like CSS, JS, or images from the static/ directory.
    Adjust the 'static/' directory path relative to this script.
    """
    static_dir = os.path.join(BASE_DIR, 'static')
    return send_from_directory(static_dir, filename)


# Entry point
if __name__ == '__main__':
    app.run(debug=True)
