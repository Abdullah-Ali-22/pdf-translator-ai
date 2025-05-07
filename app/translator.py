import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from docx import Document
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict , Type
from langchain.tools import tool

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
    temperature=0.5,
)

class TranslationsSchemaItem(BaseModel):
    """
    Schema for an individual paragraph to be translated.
    """

    header: Optional[str] = Field(
        default=None,
        description="The header of the paragraph translated into English. If there is no header, this field will be None."
    )

    content: Optional[str] = Field(
        default=None,
        description="The body content of the paragraph translated into English. If there is no content, this field will be None."
    )


class TranslationsSchema(BaseModel):
    """
    Schema for a collection of translated paragraphs.
    """

    translations: List[TranslationsSchemaItem] = Field(
        default=None,
        description="A list of translated paragraphs. Each paragraph includes an optional header and content, both translated into English."
    )


# Define Schema for Table Output
class TableTranslationSchema(BaseModel):
    table: List[Dict[str, str]] = Field(
        description="A list of dictionaries where each key-value pair represents a row and its translated columns."
    )


# Define a tool to format the translated text into a table
@tool("format_table", description="Formats translated content into a structured table.")
def format_table(data: List[TranslationsSchemaItem]) -> List[Dict[str, str]]:
    """
    Converts translated paragraphs into a structured table format.
    """
    table = []
    for item in data:
        table.append({
            "Header": item.header or "",
            "Content": item.content or ""
        })
    return table



translate_template = """
You are a highly skilled translator. Please translate the following text from German to English with precision, ensuring that the translation is complete and retains the original meaning and tone.

Please maintain the original structure by preserving the headers and their corresponding sections. Headers in the text are denoted by one or more '#' symbols (e.g., #, ##, ###, ####, #####).
- If the text is structured in a table, preserve the tabular format in the translation.

**Original Text:**
{text}
"""

#if there is any number after the header in the orginal text like # 1, # 2, ## 3 .Please include it in the translated header as well wthout '#'

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

"""def translate_text(text, is_table=False):
    result = chain.invoke({"text": text}).translations
    
    if is_table:
        # Convert result into a table format
        table_result = format_table.invoke(result)
        return TableTranslationSchema(table=table_result)
    
    return result"""