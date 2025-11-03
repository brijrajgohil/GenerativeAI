from langchain_openai import OpenAIEmbedding
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbedding(model="text-embedding-3-small", dimension=32)

documents = [
    "Delhi is the capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Paris is the capital of France."
]

results = embedding.embed_documents(documents)
print(str(results))
