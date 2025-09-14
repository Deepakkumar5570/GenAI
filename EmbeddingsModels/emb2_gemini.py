from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Make sure your .env file has GOOGLE_API_KEY=<your_api_key>
# Or you can set it manually like:
# os.environ["GOOGLE_API_KEY"] = "your_api_key_here"

# Initialize embeddings model (no dimensions argument)
emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Get embedding for a query
result = emb.embed_query("hello world")

print("Embedding length:", len(result))
print("First 10 values:", result[:10])
