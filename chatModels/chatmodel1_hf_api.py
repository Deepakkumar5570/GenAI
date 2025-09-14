from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
os.environ['HF_HOME']='F:\GenAI\models'
llm= HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"temperature":0.5, "max_length":64}
)

model= ChatHuggingFace(llm=llm)
result=model.invoke("What is AI")
print(result.content)