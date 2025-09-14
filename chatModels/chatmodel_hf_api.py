from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load enviorment variable
load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model= ChatHuggingFace(llm=llm)
result=model.invoke("What is AI")
print(result.content)

