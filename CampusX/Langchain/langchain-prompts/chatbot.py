from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()

model = ChatOpenAI()
chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)
print("Chat History: ", chat_history)