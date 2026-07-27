from langchain_openai import ChatOpenAI
from langchain_prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarise the following {text}.",
    input_variables=["text"]
)

model = ChatOpenAI()
parser = StrOutputParser()

report_gen_chain = RunnableSequence([prompt1, model, parser])
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence([prompt2, model, parser])),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

final_chain.invoke({
    "topic": "AI vs job market"
})