from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimension=300)

documents = [
    "Delhi is the capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Paris is the capital of France."
]

query = "What is the capital of India?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
print(documents[index], score)


