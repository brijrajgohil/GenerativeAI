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
    improved_story: str
    final_story: str

# Nodes
def generate_story(state: State):
    """First LLM call to generate initial story"""
    msg = llm.invoke(f"Write a short children store about {state['topic']}")
    return {"story": msg.content}

def check_mention(state: State):
    """Gate function to check if the children story has a elephant or lion"""
    # Simple check - does the joje contain ? or !
    if "elephant" in state["story"] or "lion" in state["story"]:
        return "Pass"
    return "Fail"

def improved_story(state: State):
    """Second LLM to improve the story"""
    msg = llm.invoke(f"Make this story interesting: {state['story']}")
    return {"improved_story": msg.content}

def polish_story(state: State):
    """Third LLM call for final polish"""
    msg = llm.invoke(f"Add a suprising twist to this story: {state['improved_story']}")
    return {"final_story": msg.content}

# Build a workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_story", generate_story)
workflow.add_node("improve_story", improved_story)
workflow.add_node("polish_story", polish_story)

# Add edges to connect nodes
workflow.add_edge(START, "generate_story")
workflow.add_conditional_edges(
    "generate_story", check_mention, {"Fail": "improve_story", "Pass": END}
)
workflow.add_edge("improve_story", "polish_story")
workflow.add_edge("polish_story", END)

# Compile
chain = workflow.compile()

# Show workflow
display(Image(chain.get_graph().draw_mermaid_png()))

# Invoke
state = chain.invoke({"topic": "lions"})
print("Initial story: ")
print(state["story"])
if "improved_story" in state:
    print("Improved story")
    print(state["improved_story"])

    print("Final story")
    print(state["final_story"])
else:
    print("Final story")
    print(state["story"])