from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Gemini
chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Call the model
response = chat.invoke("What is the NLP?")
print(response.content)