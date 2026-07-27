from typing import TypedDict, Literal

# Define the structure for email classification
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    # Raw email data
    email_content: str
    sender_email: str
    email_id: str

    # Classifcation result
    classification: EmailClassification | None

    # Raw search / API results
    search_results: list[str] | None # List of raw document chunks
    customer_history: dict | None # Raw customer data from CRM

    # Generated content
    draft_response: str | None
    messages: list[str] | None


# Read and classify nodes
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama

#llm = ChatOpenAI(model="gpt-5-nano")
llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0,
)

def read_email(state: EmailAgentState) -> dict:
    """Extract and parse email content"""
    # In production, this connect to your email service
    return {
        "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
    }

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """Use LLM to classify email intent and urgency, then route accordingly"""
    # Create structured LLM that returns EmailClassification dict
    structured_llm = llm.with_structured_output(EmailClassification)

    # Format the prompt on demand, not stored in state
    classification_prompt = f"""
    Analyze this customer email and classify it
    Email: {state['email_content']}

    Provide classification including intent, urgency, topic and summary
    """

    # Get structured response direct as dict
    classification = structured_llm.invoke(classification_prompt)

    # Determind next node based on classification
    if classification["intent"] == "billing" or classification["urgency"] == "critical":
        goto = "human_review"
    elif classification["intent"] in ["question", "feature"]:
        goto = "search_documentation"
    elif classification["intent"] == "bug":
            goto = "bug_tracking"
    else:
         goto = "draft_response"

    return Command(
         update={"classification": classification},
         goto=goto
    )  

# Search and tracking nodes
def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
     """Search knowledge base for relevant information"""
     # Build search query from classification
     classification = state.get("classification", {})
     query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

     try:
          # Implement your search logic here
          # Store raw search results, not formatted text
          search_results = [
               "Reset password via Settings > Security > Change Password",
               "Password must be at least 12 characters",
               "Include uppercase, lowercase, numbers, and symbols"
          ]
     except SearchAPIError as e:
          # For recoverable search errors, store error and continue
          search_results = [f"Search temporarily unavailable: str(e)"]

     return Command(
          update={"search_results": search_results},
          goto="draft_response"
     )

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
     """Create or update bug tracking ticket"""

     # Create ticket in your bug tracking system
     ticket_id ="BUG-12345"

     return Command(
          update={
               "search_results": [f"Bug ticket {ticket_id} created"],
               "current_step": bug_tracked
          },
          goto="draft_response"
     )

# Response nodes
def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
     """Generate response using context and route based on quality"""
     classification = state.get("classification", {})

     # Format context from raw state data on-demand
     context_sections = []
     if state.get("search_results"):
          formatted_docs = "\n".join([f" - {doc}" for doc in state["search_results"]])
          context_sections.append(f"Relevant documentation:\n{formatted_docs}")

     if state.get("customer_history"):
          # Format customer data for the prompt
          context_sections.append(f"Customer tier: {state['customer_history'].get('tier', 'standard')}")
     # Build the prompt with formatted context
     draft_prompt = f"""
     Draft a response to this customer email:
     {state['email_content']}
     
     Email intent: {classification.get('intent', 'unknown')}
     Urgency level: {classification.get('urgency', 'medium')}

     {chr(10).join(context_sections)}

     Guidelines:
     - Be proffessional
     - Address their specific concern
     - Use the provided documentation when relevant
     """
     response = llm.invoke(draft_prompt)

     # Determind if human review needed based on urgency and intent
     needs_review = (
          classification.get("urgency") in ["high", "critical"] or
          classification.get("intent") == "complex"
     )

     # Route to the appropriate next node
     goto = "human_review" if needs_review else "send_reply"

     return Command(
          update={"draft_response": response.content},
          goto=goto
     )

def human_review(state:EmailAgentState)-> Command[Literal["send_reply", END]]:
     """Pause for human review using interrupt and route based on decision"""
     classification = state.get("classification", {})

     # interrupt() must come first - any code before it will re-run on resume
     human_decision = interrupt({
          "email_id": state.get("email_id", ""),
          "original_email": state.get("email_content",""),
          "draft_response": state.get("draft_response", ""),
          "urgency": classification.get("urgency"),
          "intent": classification.get("intent"),
          "action": "Please review and approve/edit this response"
     })

     # Now process the human's decision
     if human_decision.get("approved"):
          return Command(
               update={"draft_response": human_decision.get("edited_response", state.get("draft_response", ""))},
               goto="send_reply"
          )
     else:
          return Command(update={}, goto=END)

def send_reply(state: EmailAgentState) -> dict:
     """Send the email response"""
     print(f"Sending reply: {state['draft_response'][:100]}...")
     return {}

# Create the agent and wire the nodes
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

# Create the graph
workflow = StateGraph(EmailAgentState)

# Add nodes with appropriate error handling
workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)

# Add retry policy for nodes that might have transient features
workflow.add_node(
     "search_documentation",
     search_documentation,
     retry_policy=RetryPolicy(max_attempts=3)
)

workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

# ADd only essential edges
workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)


memory = MemorySaver()
email_app = workflow.compile(checkpointer=memory)


# Testing the agent
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver

class EmailState(TypedDict):
     email_content: str
     response_text: str | None

def human_review_node(state: EmailState):
     interrupt(
          {
               "approved": False,
               "edited_response": state.get("response_text") or "",
          }
     )
     return {"response_text": "placeholder"}


review_test_app = (
     StateGraph(EmailState)
     .add_node("human_review", human_review_node)
     .add_edge(START, "human_review")
     .add_edge("human_review", END)
     .compile(checkpointer=InMemorySaver())
)

initial_state = {
    "email_content": "I was charged twice for my subscription! This is urgent!",
    "response_text": "Draft response",
}

# Run with a thread_id for persistence
config = {"configurable": {"thread_id": "customer_123"}}
stream = review_test_app.stream_events(initial_state, config, version="v3")
_ = stream.output  # drive the stream to completion
# The graph will pause at human_review
print(f"human review interrupt:{stream.interrupts}")

human_response = Command(
    resume={
        "approved": True,
        "edited_response": "We sincerely apologize for the double charge. I've initiated an immediate refund...",
    }
)

# Resume execution
resumed = review_test_app.stream_events(human_response, config, version="v3")
final_state = resumed.output
print("Email sent successfully!")
