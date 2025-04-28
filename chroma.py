import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Step 1: Extract PDF text
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

# Step 2: Chunk text
text = extract_text_from_pdf("Swapnil_resume_kd.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# Step 3: Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()

# Step 4: Store in Chroma
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("pdf-doc")

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk-{i}" for i in range(len(chunks))]
)



def query_chroma(query , collection , model , top_k = 5):

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings = query_embedding,
        n_results = top_k
    )

    return results["documents"], results["distances"]

query = "What is Swapnil's experience with machine learning?"
documents , distances = query_chroma(query , collection , model)

for doc , dist in zip(documents , distances):
    print(f"Document: {doc}")
    print(f"Similarity Score: {dist}")  # The lower the score, the better the match
    print("---")