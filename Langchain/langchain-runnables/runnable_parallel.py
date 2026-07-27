from langchain_openai import ChatOpenAI
from langchain_prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a tweet about a {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about a {topic}.",
    input_variables=["topic"]
)
model = ChatOpenAI()
parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'tweet': RunnableSequence([prompt1, model, parser]),
    'linkedin': RunnableSequence([prompt2, model, parser])
})

result = parallel_chain.invoke({"topic": "new AI technology"})
print(result)
print(result['tweet'])
print(result['linkedin'])