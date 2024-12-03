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
    temperature=0.0,
)


class JobDescriptionSchema(BaseModel):
    """
    Schema for the job description details.
    """
    must: Optional[List[str]] = Field(default=None, description="List of required skills and experience (Must)")
    target: Optional[List[str]] = Field(default=None, description="List of target skills and experience (Target)")

class JobDetailsSchema(BaseModel):
    """
    Schema for the job details table.
    """
    role_position: Optional[str] = Field(default=None, description="The role or position name")
    location: Optional[str] = Field(default=None, description="Job location")
    number_of_fte: Optional[int] = Field(default=None, description="Number of full-time equivalents required")
    rgs_id: Optional[str] = Field(default=None, description="RGS ID")
    remote_onsite: Optional[str] = Field(default=None, description="Remote or onsite details with specific distribution")
    onsite_frequency_week: Optional[str] = Field(default=None, description="Frequency of onsite work per week")
    project_duration: Optional[str] = Field(default=None, description="Duration of the project")
    working_hours_per_day: Optional[str] = Field(default=None, description="Number of working hours per day")
    contract_mode: Optional[str] = Field(default=None, description="Contract type, e.g., Freelancer or Employee")
    daily_rate: Optional[str] = Field(default=None, description="Daily rate in euros")
    language_proficiency: Optional[str] = Field(default=None, description="Required language proficiency")
    start_date_of_engagement: Optional[str] = Field(default=None, description="Start date of engagement")
    experience_required: Optional[str] = Field(default=None, description="Years and type of experience required")
    job_description: Optional[JobDescriptionSchema] = Field(default=None, description="Job description details including Must and Target fields")



JobDetails_filling_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in extracting job details. "
            "fill the relevant fields from the given schema. "
            "If a answer is not present, return null for the attribute's value.",
        ),
        MessagesPlaceholder("examples"),
        ("human", "{text}"),
    ]
)



JobDetails_filling_prompt.invoke(
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
    ('We are looking for a Software Engineer based in Berlin. The role is remote with 2 days onsite per week. The project duration is 6 months, with 8 working hours per day. The contract mode is Freelancer with a daily rate of 500 euros. The candidate must be proficient in English and have at least 3 years of experience in software development. Required skills include Python and Django, while React and Node.js are desirable.',
      JobDetailsSchema(role_position="Software Engineer", location="Berlin", number_of_fte=1, rgs_id="1234", remote_onsite="Remote", onsite_frequency_week="2", project_duration="6 months", working_hours_per_day="8", contract_mode="Freelancer", daily_rate="500", language_proficiency="English", start_date_of_engagement="2023-01-01", experience_required="3 years of experience in software development", job_description=JobDescriptionSchema(must=["Python", "Django"], target=["React", "Node.js"]))),
     ]
messages = []

for text, tool_call in examples:
    messages.extend(
        tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
    )


runnable = JobDetails_filling_prompt | llm.with_structured_output(
        schema=JobDetailsSchema,
        method="function_calling",
        include_raw=False,
    )


def extract_job_details(text: str) -> JobDetailsSchema:
        """
        Function to call the runnable and return the extracted job details.
        
        Args:
            text (str): The input text containing job details.
            
        Returns:
            JobDetailsSchema: The extracted job details.
        """
        response = runnable.invoke({"text": text, "examples": messages})
        return response


