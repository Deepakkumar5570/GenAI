from langchain_huggingface import HuggingFaceEmbeddings

emb= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

text="My Name Is Deepak"
vector = emb.embed_query(text)
print("Single text embedding (length):", len(vector))
print(vector[:10])