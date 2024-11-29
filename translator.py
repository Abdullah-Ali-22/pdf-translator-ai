import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from docx import Document
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict , Type

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
    temperature=0,
)


class translationsschema(BaseModel):
    """
    Schema for a paragraph to translate.
    """

    header: Optional[str] = Field(default=None, description="The header of the paragraph translated to English")
    content: Optional[str] = Field(default=None, description="The content of the paragraph translated to English")

class TranslationsSchema(BaseModel):
    """
    lIST OF Translations schema
    """
    translations: List[translationsschema] = Field(default=None, description="List of translations")





translate_template = """
You are a highly skilled translator. Please translate the following text from German to english, ensuring that the translation is accurate, full and maintains the original meaning and tone:

example_input:




given_text: {text}
"""


structured_llm = llm.with_structured_output(TranslationsSchema)


translate_prompt = PromptTemplate.from_template(translate_template)

chain = (
    translate_prompt | structured_llm )

def translate_text(text):
    result = chain.invoke({"text": text})
    #tranlated_header= result.header
    #tranlated_content= result.content
    #return tranlated_header, tranlated_content
    return result.translations

