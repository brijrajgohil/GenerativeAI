from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_ollama import ChatOllama


# model = init_chat_model(
#     "claude-sonnet-4-6",
#     temperature=0
# )

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)

# Graph State
class State(TypedDict):
    topic: str
    story: str
    joke: str
    poem: str
    combined_output: str


# Nodes
def generate_story(state: State):
    """First LLM call to generate story"""
    msg = llm.invoke(f"Write a short children story about {state['topic']}")
    return {"story": msg.content}


def generate_joke(state: State):
    """First LLM call to generate joke"""
    msg = llm.invoke(f"Write a short children joke about {state['topic']}")
    return {"joke": msg.content}

def generate_poem(state: State):
    """First LLM call to generate poem"""
    msg = llm.invoke(f"Write a short children peom about {state['topic']}")
    return {"poem": msg.content}

def aggregator(state: State):
    """Combine the joke, story and poem into a single output"""
    combined = f"Heres a story, joke and poem about {state['topic']}\n\n"
    combined += f"STORY: \n{state['story']}\n\n"
    combined += f"JOKE: \n{state['joke']}\n\n"
    combined += f"POEM: \n{state['poem']}\n\n"
    return {"combined_output": combined}

# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("generate_story", generate_story)
parallel_builder.add_node("generate_joke", generate_joke)
parallel_builder.add_node("generate_poem", generate_poem)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodesl
parallel_builder.add_edge(START, "generate_story")
parallel_builder.add_edge(START, "generate_joke")
parallel_builder.add_edge(START, "generate_poem")
parallel_builder.add_edge("generate_story", "aggregator")
parallel_builder.add_edge("generate_joke", "aggregator")
parallel_builder.add_edge("generate_poem", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Invoke
state = parallel_workflow.invoke({"topic": "cats"})
print(state["combined_output"])