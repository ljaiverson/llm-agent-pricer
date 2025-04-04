# Imports
import os
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')

# Database path
DB = "products_vectorstore"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=DB)

# Check if the collection exists and delete it if it does
collection_name = "products"
existing_collection_names = [collection.name for collection in client.list_collections()]
if collection_name in existing_collection_names:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")

# Create a new collection
collection = client.create_collection(collection_name)

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load training data from pickle file
with open('train.pkl', 'rb') as file:
    train = pickle.load(file)

# Function to extract description from an item
def description(item):
    # Remove the question prompt and extract the description text
    text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
    return text.split("\n\nPrice is $")[0]

# Add documents, embeddings, and metadata to the Chroma collection in batches
for i in tqdm(range(0, len(train), 1000)):
    documents = [description(item) for item in train[i: i+1000]]
    vectors = model.encode(documents).astype(float).tolist()
    metadatas = [{"category": item.category, "price": item.price} for item in train[i: i+1000]]
    ids = [f"doc_{j}" for j in range(i, i+1000)]
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas
    )
