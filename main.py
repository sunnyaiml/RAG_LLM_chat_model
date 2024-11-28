from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Load documents (PDF or TXT)
def load_document(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
        documents = loader.load()
    else:
        raise ValueError("Unsupported file format. Use PDF or TXT.")
    return documents

# Split text into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Create embeddings and store in FAISS
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# Create QA chain with Hugging Face model
def qa_chain(vector_store):
    retriever = vector_store.as_retriever()
    
    # Load Hugging Face model with adjusted parameters for max_length and truncation
    hf_pipeline = pipeline('text-generation', model='gpt2', max_length=500, truncation=True)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # Set up the QA chain using the LLM and retriever
    return RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

# User interaction
def main():
    file_path = input("Enter the path to your PDF or TXT file: ")
    documents = load_document(file_path)
    chunks = split_text(documents)
    vector_store = create_vector_store(chunks)
    qa = qa_chain(vector_store)
    
    # Ask questions
    while True:
        question = input("\nAsk a question about the document (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = qa.run(question)  # Using run() instead of invoke()
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
