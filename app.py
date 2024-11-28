
from flask import Flask, request, jsonify
import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA 
import shutil

app = Flask(__name__)

# Path to store uploaded files and vector store
UPLOAD_FOLDER = 'uploads'
VECTOR_STORE_PATH = 'faiss_index'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to hold the vector store and the retriever
vector_store = None
qa_chain = None



# Function to create embeddings for the document
def get_document_embeddings(documents):
    # Example: Replace this with your embedding logic, e.g., using HuggingFace or another model.
    # Here, we're assuming that each document is a string and embedding is a list of floats.
    embeddings = []  # Your logic to generate embeddings for documents
    for doc in documents:
        # Generate embedding for the document
        embedding = [0.0] * 512  # Placeholder for the actual embedding
        embeddings.append(embedding)
    return embeddings

# Function to save the embeddings and index to a pickle file
def save_model_to_pickle(documents, embeddings, filename):
    # Create a FAISS index (using L2 distance as an example)
    dim = len(embeddings[0])  # assuming all embeddings have the same dimension
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save the FAISS index to a file
    faiss.write_index(index, f"{filename}_index.index")

    # Save the documents to a pickle file
    with open(f"{filename}_documents.pkl", 'wb') as f:
        pickle.dump(documents, f)



# Helper function to load the document
def load_document(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
        documents = loader.load()
    else:
        raise ValueError("Unsupported file format. Only PDF and TXT are allowed.")
    return documents

# Helper function to split the document into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Helper function to create vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# Route to upload PDF/TXT file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save file to upload folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    return jsonify({"message": f"File {file.filename} uploaded successfully."}), 200

# Route to train the model using uploaded file
@app.route('/trained', methods=['POST'])
def train_model():
    global vector_store, qa_chain

    # Check if a file has been uploaded
    files = os.listdir(UPLOAD_FOLDER)
    if len(files) == 0:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, files[0])
    print(file_path)
    try:
        documents = load_document(file_path)
        print(documents)
        chunks = split_text(documents)
        vector_store = create_vector_store(chunks)

        # Save the FAISS index for reuse
        vector_store.save_local(VECTOR_STORE_PATH)

        # Initialize Hugging Face model and pipeline
        hf_pipeline = pipeline('text-generation', model='gpt2', max_length=500, truncation=True)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
        print("hf",hf_pipeline)
        # Create the retrieval QA chain
        retriever = vector_store.as_retriever()
        print("retriever",retriever)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

        return jsonify({"message": "Model trained successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to ask questions to the trained model
@app.route('/ask', methods=['POST'])
def ask_question():
    if not qa_chain:
        return jsonify({"error": "Model is not trained yet. Please train the model first."}), 400
    
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided."}), 400
    
    try:
        # Get the answer from the trained model
        response = qa_chain.invoke(question)
        
        # Splitting the response into an array of answers by newlines or other delimiters
        answers = response.split("\n")
        
        # Clean up empty answers or context-based information (optional)
        answers = [answer.strip() for answer in answers if answer.strip()]
        
        return jsonify({"answers": answers}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
