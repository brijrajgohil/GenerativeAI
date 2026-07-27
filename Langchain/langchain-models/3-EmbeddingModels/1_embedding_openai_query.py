from langchain_openai import OpenAIEmbedding
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbedding(model="text-embedding-3-small", dimension=32)
result = embedding.embed_query("Delhi is the capital India")
print(str(result))
