import os
import uuid
import json
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from docx import Document
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict , Type
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
# Load environment variables
from dotenv import load_dotenv
load_dotenv()




# Set up LLM configuration
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-10-01-preview",
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    temperature=0.3,
)


class PDFSummarySchema(BaseModel):
    """
    Schema for extracting a title and summary from a PDF document.
    """
    title: Optional[str] = Field(default=None, description="The title of the document or main subject")
    summary: Optional[str] = Field(default=None, description="A concise summary of the document content")


PDFSummary_filling_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in extracting key information from documents. "
            "Extract the title and provide a concise summary of the document. "
            "If information is not present, return null for that field.",
        ),
        MessagesPlaceholder("examples"),
        ("human", "{text}"),
    ]
)


PDFSummary_filling_prompt.invoke(
    {"text": "this is some text", "examples": [HumanMessage(content="testing 1 2 3")]}
)

class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages


examples = [
    ('Quality Test Engineer job posting for a junior position. This role involves ensuring product quality through systematic testing and quality assurance processes. The candidate will be responsible for creating test plans, executing tests, and documenting results. They should have basic knowledge of testing methodologies and quality standards. Experience with automated testing tools is a plus. The position requires good attention to detail and analytical skills.',
      PDFSummarySchema(title="Quality Test Engineer - Junior Position", summary="A junior-level Quality Test Engineer role focused on quality assurance and testing. The position involves creating and executing test plans, documenting results, and requires knowledge of testing methodologies. The ideal candidate will have good attention to detail, analytical skills, and preferably experience with automated testing tools.")),
     ]
messages = []

for text, tool_call in examples:
    messages.extend(
        tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
    )


runnable = PDFSummary_filling_prompt | llm.with_structured_output(
        schema=PDFSummarySchema,
        method="function_calling",
        include_raw=False,
    )


def extract_pdf_summary(text: str) -> PDFSummarySchema:
        """
        Function to extract a title and summary from document text.
        
        Args:
            text (str): The input text from the PDF document.
            
        Returns:
            PDFSummarySchema: The extracted title and summary.
        """
        response = runnable.invoke({"text": text, "examples": messages})
        return response


