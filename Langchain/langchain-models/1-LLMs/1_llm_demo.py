from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model_name="gpt3.5-turbo-instruct")

result = llm.invoke("What is the capital of India?")

print(result)