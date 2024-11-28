import os
import pickle
import logging
import re  # Make sure to import re module

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
app = Flask(__name__)
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

        prompt_template = """Use the following context question the answer is this.
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
    return render_template("index.html")

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

        # Extract the helpful answer section
        match = re.search(r'Helpful Answer:(.*)', response, re.DOTALL)
        if match:
            helpful_answer = match.group(1).strip()
        else:
            helpful_answer = "No helpful answer found."

        # Split into answers if multiple lines
        answers = [ans.strip() for ans in helpful_answer.split('\n') if ans.strip()]

        return jsonify({
            "answers": answers,
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





# from flask import Flask, request, jsonify
# import os
# from langchain.document_loaders import PyPDFLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
# from transformers import pipeline
# from langchain.chains import RetrievalQA 
# import shutil

# app = Flask(__name__)

# # Path to store uploaded files and vector store
# UPLOAD_FOLDER = 'uploads'
# VECTOR_STORE_PATH = 'faiss_index'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Global variable to hold the vector store and the retriever
# vector_store = None
# qa_chain = None



# # Function to create embeddings for the document
# def get_document_embeddings(documents):
#     # Example: Replace this with your embedding logic, e.g., using HuggingFace or another model.
#     # Here, we're assuming that each document is a string and embedding is a list of floats.
#     embeddings = []  # Your logic to generate embeddings for documents
#     for doc in documents:
#         # Generate embedding for the document
#         embedding = [0.0] * 512  # Placeholder for the actual embedding
#         embeddings.append(embedding)
#     return embeddings

# # Function to save the embeddings and index to a pickle file
# def save_model_to_pickle(documents, embeddings, filename):
#     # Create a FAISS index (using L2 distance as an example)
#     dim = len(embeddings[0])  # assuming all embeddings have the same dimension
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)

#     # Save the FAISS index to a file
#     faiss.write_index(index, f"{filename}_index.index")

#     # Save the documents to a pickle file
#     with open(f"{filename}_documents.pkl", 'wb') as f:
#         pickle.dump(documents, f)



# # Helper function to load the document
# def load_document(file_path):
#     if file_path.endswith('.pdf'):
#         loader = PyPDFLoader(file_path)
#         documents = loader.load()
#     elif file_path.endswith('.txt'):
#         loader = TextLoader(file_path)
#         documents = loader.load()
#     else:
#         raise ValueError("Unsupported file format. Only PDF and TXT are allowed.")
#     return documents

# # Helper function to split the document into chunks
# def split_text(documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     return text_splitter.split_documents(documents)

# # Helper function to create vector store
# def create_vector_store(chunks):
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     vector_store = FAISS.from_documents(chunks, embeddings)
#     return vector_store

# # Route to upload PDF/TXT file
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     # Save file to upload folder
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)
    
#     return jsonify({"message": f"File {file.filename} uploaded successfully."}), 200

# # Route to train the model using uploaded file
# @app.route('/trained', methods=['POST'])
# def train_model():
#     global vector_store, qa_chain

#     # Check if a file has been uploaded
#     files = os.listdir(UPLOAD_FOLDER)
#     if len(files) == 0:
#         return jsonify({"error": "No file uploaded"}), 400

#     file_path = os.path.join(UPLOAD_FOLDER, files[0])
#     print(file_path)
#     try:
#         documents = load_document(file_path)
#         print(documents)
#         chunks = split_text(documents)
#         vector_store = create_vector_store(chunks)

#         # Save the FAISS index for reuse
#         vector_store.save_local(VECTOR_STORE_PATH)

#         # Initialize Hugging Face model and pipeline
#         hf_pipeline = pipeline('text-generation', model='gpt2', max_length=500, truncation=True)
#         llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
#         print("hf",hf_pipeline)
#         # Create the retrieval QA chain
#         retriever = vector_store.as_retriever()
#         print("retriever",retriever)
#         qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

#         return jsonify({"message": "Model trained successfully."}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Route to ask questions to the trained model
# @app.route('/ask', methods=['POST'])
# def ask_question():
#     if not qa_chain:
#         return jsonify({"error": "Model is not trained yet. Please train the model first."}), 400
    
#     question = request.json.get('question')
#     if not question:
#         return jsonify({"error": "No question provided."}), 400
    
#     try:
#         # Get the answer from the trained model
#         response = qa_chain.invoke(question)
        
#         # Splitting the response into an array of answers by newlines or other delimiters
#         answers = response.split("\n")
        
#         # Clean up empty answers or context-based information (optional)
#         answers = [answer.strip() for answer in answers if answer.strip()]
        
#         return jsonify({"answers": answers}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
