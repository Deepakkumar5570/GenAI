# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv# Install dependencies (run in terminal if not installed already)
# pip install langchain-google-genai google-generativeai

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Set your Google API key (replace with your actual key)
# os.environ["GOOGLE_API_KEY"] = "your_api_key_here"

# 2. Initialize the embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 3. Create embeddings for a single text
text = "LangChain makes it easy to build applications with LLMs."
vector = embeddings.embed_query(text)
print("Single text embedding (length):", len(vector))
print(vector[:10])  # show first 10 numbers

# 4. Create embeddings for multiple texts
texts = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "Large language models power chatbots."
]
vectors = embeddings.embed_documents(texts)

print("\nMultiple text embeddings:")
for i, v in enumerate(vectors):
    print(f"Text {i+1} embedding length: {len(v)}")
