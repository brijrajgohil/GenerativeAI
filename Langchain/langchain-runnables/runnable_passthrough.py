from langchain_openai import ChatOpenAI
from langchain_prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

def word_count(text: str) -> int:
    return len(text.split())


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

joke_gen_chain = RunnableSequence([prompt, model, parser])
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count),
})

final_chain = RunnableSequence([joke_gen_chain, parallel_chain])

result = final_chain.invoke({"topic": "computers"})
print(result)