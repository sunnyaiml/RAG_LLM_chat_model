import os
import pickle
import logging
import re
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Initialize Flask app and enable CORS
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# Define file paths
UPLOAD_FOLDER = 'uploads'
PICKLE_FOLDER = 'pickle'
VECTOR_STORE_PATH = 'faiss_index'

# Create necessary folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PICKLE_FOLDER, exist_ok=True)

# Global variables to hold the vector store and QA chain
vector_store = None
qa_chain = None

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

def save_qa_chain_to_pickle(qa_chain, file_name='qa_chain.pkl'):
    """Save the trained QA chain (including retriever and LLM) to a pickle file."""
    try:
        with open(os.path.join(PICKLE_FOLDER, file_name), 'wb') as file:
            pickle.dump(qa_chain, file)
        logging.info(f"QA chain successfully saved to {file_name}")
    except Exception as e:
        logging.error(f"Error while saving QA chain to pickle: {e}")

def load_qa_chain_from_pickle(file_name='qa_chain.pkl'):
    """Load the trained QA chain (including retriever and LLM) from a pickle file."""
    global qa_chain
    try:
        pickle_path = os.path.join(PICKLE_FOLDER, file_name)
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as file:
                qa_chain = pickle.load(file)
            logging.info(f"QA chain successfully loaded from {file_name}")
            return qa_chain
        else:
            logging.warning(f"{file_name} not found in pickle folder.")
    except Exception as e:
        logging.error(f"Error loading QA chain from pickle: {e}")
    return None

def load_document(file_path):
    """Load document based on file type (PDF or TXT)."""
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        else:
            raise ValueError("Unsupported file format. Only PDF and TXT are allowed.")
        logging.info(f"Document loaded from {file_path}")
        return documents
    except Exception as e:
        logging.error(f"Error loading document: {e}")
        raise

def split_text(documents):
    """Split documents into smaller chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Document split into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        raise

def create_vector_store(chunks):
    """Create a vector store from document chunks."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_store = FAISS.from_documents(chunks, embeddings)
        logging.info("Vector store created successfully.")
        return vector_store
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        return None

def create_custom_qa_chain(vector_store):
    """
    Create a custom QA chain with a specific prompt.
    """
    try:
        hf_pipeline = pipeline(
            'text-generation',
            model='gpt2',
            max_length=500,
            do_sample=True,
            top_k=50,
            temperature=0.7  # Set the temperature to a positive float value
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        prompt_template = """Use the following context to answer the question.
        Context: {context}
        Question: {question}
        Helpful Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            chain_type_kwargs={'prompt': PROMPT}
        )
        logging.info("Custom QA chain created successfully.")
        return qa_chain
    except Exception as e:
        logging.error(f"Error creating QA chain: {e}")
        return None

@app.route('/')
def home():
    return render_template("index.html", time=time)

@app.route('/chat')
def chat():
    return render_template('chat.html', time=time)

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    """Upload PDF or TXT file."""
    try:
        if 'file' not in request.files:
            logging.error("No file part in the request")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            logging.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        # Print file details for debugging
        logging.info(f"Received file: {file.filename}")
        logging.info(f"File content type: {file.content_type}")

        # Clean up previous uploads
        for existing_file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, existing_file))

        # Save file to upload folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        logging.info(f"File saved successfully at {file_path}")
        return jsonify({"message": f"File {file.filename} uploaded successfully."}), 200

    except Exception as e:
        logging.error(f"Error in file upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/trained', methods=['POST'])
def train_model():
    """Train the model using the uploaded file."""
    global vector_store, qa_chain

    # Check if a file has been uploaded
    files = os.listdir(UPLOAD_FOLDER)
    if len(files) == 0:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, files[0])
    try:
        # Load and process document
        documents = load_document(file_path)
        chunks = split_text(documents)

        # Create vector store
        vector_store = create_vector_store(chunks)
        if vector_store is None:
            return jsonify({"error": "Failed to create vector store."}), 500

        # Create QA chain
        qa_chain = create_custom_qa_chain(vector_store)

        # Save QA chain to pickle
        save_qa_chain_to_pickle(qa_chain)

        return jsonify({"message": "Model trained successfully."}), 200
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask a question to the trained model."""
    global qa_chain

    # Ensure model is trained
    if qa_chain is None:
        # Try to load from pickle
        qa_chain = load_qa_chain_from_pickle()
        if qa_chain is None:
            return jsonify({"error": "Model is not trained yet. Please train the model first."}), 400

    # Get question from request
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        # Get the answer from the trained model
        response = qa_chain.run(question)

        # Ensure response is a string
        if not isinstance(response, str):
            response = str(response)

        # Extract the helpful answer section or use the full response
        match = re.search(r'Helpful Answer:(.*)', response, re.DOTALL)
        if match:
            helpful_answer = match.group(1).strip()
        else:
            helpful_answer = response.strip()  # Use the full response if no helpful answer found

        # Return the entire helpful answer or full response as a single response
        return jsonify({
            "answer": helpful_answer,
            "original_response": response
        }), 200
    except Exception as e:
        logging.error(f"Error during question answering: {e}")
        return jsonify({
            "error": str(e),
            "details": repr(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
