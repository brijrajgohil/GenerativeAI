from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI()

class Review(TypedDict):
    key_themes: Annotated[list[str], "Key themes discussed in the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "The sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "List of pros mentioned in the review"]

structured_model = model.with_structured_output(Review)
result = structured_model.invoke("The hardware is great, but software is buggy.")
print(result)
print(result['summary'])
print(result['sentiment'])
