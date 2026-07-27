from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()
model2 = ChatAnthropic()

prompt1 = PromptTemplate(
    template="Generate short an simplee notes for the following text \n {text}.",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 short question answer from the following text \n {text}.",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and QA into a single document \n notes ->{notes} and QA -> {qa}",
    input_variables=["notes", "qa"]
)

parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser
chain = parallel_chain | merge_chain
result = chain.invoke({"text": "LangChain is a framework for developing applications powered by language models. It enables developers to build complex applications by chaining together various components such as prompts, models, and output parsers. LangChain supports multiple language model providers, allowing for flexibility and choice in model selection. The framework also includes tools for prompt management, output parsing, and integration with external data sources, making it easier to create sophisticated AI-driven applications."})
print(result)
chain.get_graph().print_ascii()

