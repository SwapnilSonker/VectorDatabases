import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import weaviate
# from weaviate.util import generate_uuid5
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.init import Auth
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

CLASS_NAME = "PDFChunk"
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")


def generate_uuid5(chunk):
    return str(uuid.uuid4())

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

text = extract_text_from_pdf("Swapnil_resume_kd.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# Step 3: Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()

client = weaviate.connect_to_weaviate_cloud(
    cluster_url = WEAVIATE_URL,
    auth_credentials = Auth.api_key(WEAVIATE_API_KEY),
    skip_init_checks=True
)

# print(client.is_ready())

# client.close()

# === 5. Create schema (if not exists) ===
if CLASS_NAME not in client.collections.list_all():
    client.collections.create(
        name=CLASS_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="text", data_type=DataType.TEXT)
        ]
    )

# === 6. Insert data ===
collection = client.collections.get(CLASS_NAME)

for chunk, vector in zip(chunks, embeddings):
    collection.data.insert(
        properties={"text": chunk},
        vector=vector,
        uuid=generate_uuid5(chunk)
    )

print("✅ Uploaded all chunks to Weaviate!")

# === 7. Query example ===
query = "Experience with NLP ?"
query_vector = model.encode([query])[0]

results = collection.query.near_vector(
    near_vector=query_vector,
    limit=3
)

# text_results = collection.query.near_text(
#     near_text = {"concepts" : [query]},
#     limit = 5,
# )

# print("\n Top results")
# for obj in results.objects:
#     print("near text --" , obj.properties["text"])

print("\n🔍 Top Matches:")
for obj in results.objects:
    print("-", obj.properties["text"])

# === 8. Close connection ===
client.close()