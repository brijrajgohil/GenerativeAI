from langchain_openai import ChatOpenAI
from langchain_prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="Tell me a joke about {topic}.",
    input_variables=["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}.",
    input_variables=["text"]
)

chain = RunnableSequence([prompt, model, parser, prompt2, model, parser])

result = chain.invoke({"topic": "computers"})
print(result)