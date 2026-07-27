from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

class Review(BaseModel):
    key_themese: list[str] = Field(description="Key themes discussed in the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="The sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="List of pros mentioned in the review")
    cons: Optional[list[str]] = Field(default=None, description="List of cons mentioned in the review")

structured_model = model.with_structured_output(Review)
result = structured_model.invoke("The hardware is great, but software is buggy.")
print(result)
print(result['summary'])
print(result['sentiment'])
