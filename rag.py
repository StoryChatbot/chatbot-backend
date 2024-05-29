import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

file = "./data/chi pdf fih l9isas.pdf"
folder_path = "./vectordb"

# Read the PDF file
pdf = PyPDF2.PdfReader(file)
pdf_text = ""
for page in pdf.pages:
    pdf_text += page.extract_text()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
chunks = text_splitter.split_text(pdf_text)
print("Chunks")

# Create a Chroma vector store
embeddings = OllamaEmbeddings(model="smit lmodel dyalk")
print("Embedding")

vector_store = Chroma.from_texts(
    texts=chunks, embedding=embeddings, persist_directory=folder_path
)
print("vector_store")

vector_store.persist()
print("The data is persisted successfully ")