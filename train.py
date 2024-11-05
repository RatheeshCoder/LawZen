from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain.embeddings.base import Embeddings
import os

print("Step 1: Importing libraries completed")

# Wrapper class to make HuggingFaceEmbedding compatible with LangChain
class LangChainHuggingFaceEmbeddings(Embeddings):
    def __init__(self, huggingface_embedding):
        self.huggingface_embedding = huggingface_embedding

    def embed_documents(self, texts):
        return self.huggingface_embedding.get_text_embedding_batch(texts)

    def embed_query(self, text):
        return self.huggingface_embedding.get_text_embedding(text)

# Load all PDF files from the 'data' directory
loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)
print("Step 2: Directory loader initialized")

# Load all documents
documents = loader.load()
print(f"Step 3: {len(documents)} documents loaded")

# Process the full content of each document
print("Step 4: Processing full content of documents")

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Step 5: Documents split into {len(texts)} text chunks")

# Initialize the embeddings
hf_embedding = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = LangChainHuggingFaceEmbeddings(hf_embedding)
print("Step 6: Embeddings initialized")

# Set up the Chroma vector store
persist_directory = "trained"
db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
print("Step 7: Chroma vector store created and data persisted")

print("Process completed successfully")