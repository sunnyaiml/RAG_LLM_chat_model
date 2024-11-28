
# Document-Based RAG LLM Chat Model

Welcome to the Document-Based Retrieval-Augmented Generation (RAG) Chat Model! This application allows users to upload PDF or text documents, ask questions about their content, and receive insightful answers powered by a custom-trained language model. It's designed using Flask and integrates with HuggingFace's models and FAISS for efficient retrieval.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Document Upload:** Supports both PDF and TXT file uploads.
- **Custom LLM QA Model:** Utilizes HuggingFace Transformers for generating answers.
- **Vector Store:** FAISS-based indexing ensures efficient document retrieval.
- **Persistent Storage:** Saves trained QA models using `pickle` for faster reloads.
- **REST API:** Provides clear endpoints for document handling and querying.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sunnyaiml/RAG_LLM_chat_model.git
cd RAG_LLM_chat_model
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Ensure Required Directories Exist

The script automatically creates necessary folders (uploads, pickle), but you can manually create them:

```bash
mkdir uploads pickle
```

---

## Usage

### Start the Server

```bash
python app.py
```

The Flask app will run on http://localhost:5000.

### Access the Application

Open http://localhost:5000 in your browser to use the interface.

---

## Endpoints

1. **Home Page (/)**
   - Displays the index page.

2. **Upload File (/upload)**
   - Method: POST
   - Description: Uploads a PDF or TXT document.
   - Parameters: Form data with the file.
   - Response: Confirmation message on successful upload.

3. **Train Model (/trained)**
   - Method: POST
   - Description: Processes the uploaded document, creates embeddings, and trains the QA model.
   - Response: Message indicating training success.

4. **Ask a Question (/ask)**
   - Method: POST
   - Description: Queries the trained model with a user question.
   - Parameters: JSON with the question (`{"question": "Your question"}`).

   - Response: JSON with the answers.

---

## Project Structure

```bash
RAG_LLM_chat_model/
│
├── uploads/                   # Uploaded documents
├── pickle/                    # Pickled QA chain for persistence
├── app.py                     # Main Flask application
├── requirements.txt           # List of dependencies
└── templates/
    └── index.html             # Main web page
```

---

## Dependencies

- Python 3.8+
- Flask
- Flask-CORS
- HuggingFace Transformers
- FAISS
- PyPDFLoader
- langchain

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Future Enhancements

- Support for more document formats (e.g., DOCX).
- Enhanced LLM models for more accurate responses.
- User authentication and session management.
- Deployment in production environments with Docker.

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change. Ensure your code follows the existing style guidelines and passes all tests.

---

Happy coding! 🚀

---

## Code Breakdown

### Import Statements

The script begins with necessary imports, including Flask for web handling, langchain for text processing, and HuggingFace for leveraging pre-trained language models.

### Flask Initialization and Configuration

The Flask application is initialized, and CORS (Cross-Origin Resource Sharing) is enabled to allow cross-origin requests.

### Global Variables and Directory Setup

Defines directories for uploads and storing pickled models. It also ensures these directories are created if they don't exist.

### Utility Functions

- `save_qa_chain_to_pickle`: Saves the trained QA chain to a pickle file for persistence.
- `load_qa_chain_from_pickle`: Loads the QA chain from a pickle file if it exists.
- `load_document`: Loads a document based on its file type (PDF or TXT).
- `split_text`: Splits documents into smaller chunks using `RecursiveCharacterTextSplitter`.
- `create_vector_store`: Creates a vector store from document chunks using FAISS and HuggingFaceEmbeddings.
- `create_custom_qa_chain`: Creates a custom QA chain with a specific prompt template.

### Flask Routes

- **Home Route (/):** Renders the `index.html` page.
- **Upload Route (/upload):** Handles file uploads, saves the file to the `uploads` directory, and cleans up previous uploads.
- **Train Route (/trained):** Processes the uploaded document, splits it into chunks, creates a vector store, and trains the QA model.
- **Ask Route (/ask):** Accepts a question as JSON input, queries the trained model, and returns the answer.

### Main Function

The script runs the Flask app in debug mode, making it easy to test and develop locally.
