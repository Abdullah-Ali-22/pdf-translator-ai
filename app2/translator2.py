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
    temperature=0.9,
)

class TranslationsSchemaItem(BaseModel):  
    """  
    Schema for an individual translated paragraph.  
  
    Attributes:  
        header (Optional[str]): The header of the paragraph translated into English.   
                                This captures any heading or title that precedes the paragraph.  
                                If the original paragraph does not contain a header, this field will be None.  
        content (Optional[str]): The body content of the paragraph translated into English.   
                                 This is the main text of the paragraph. If there is no content, this field will be None.  
        header_level (Optional[int]): The level of the header (e.g., 1 for main headers, 2 for sub-headers).  
                                      This helps in structuring the document with appropriate heading styles.  
        list_type (Optional[str]): The type of list for content, if applicable (e.g., 'bullet', 'numbered').  
                                   Useful for maintaining list formatting.  
        table_format (Optional[bool]): Indicates if the content is part of a table structure.   
                                       This helps in applying table formatting in the Word document.  
        notes (Optional[str]): Any additional notes or comments related to the paragraph.   
                               Useful for adding context or metadata that might not be part of the main content.  
    """  
    header: Optional[str] = Field(  
        default=None,  
        description=("The header of the paragraph translated into English. "  
                     "This is used to capture any heading or title that precedes the paragraph. "  
                     "If the original paragraph does not contain a header, this field will be None.")  
    )  
    content: Optional[str] = Field(  
        default=None,  
        description=("The body content of the paragraph translated into English. "  
                     "This includes the main text or message of the paragraph. "  
                     "If the original paragraph does not contain any content, this field will be None.")  
    )  
    header_level: Optional[int] = Field(  
        default=None,  
        description=("The level of the header, indicating its position in the document hierarchy. "  
                     "For example, 1 for main headers, 2 for sub-headers.")  
    )  
    list_type: Optional[str] = Field(  
        default=None,  
        description=("The type of list for content, if applicable. "  
                     "Indicates whether the content is part of a 'bullet' or 'numbered' list.")  
    )  
    table_format: Optional[bool] = Field(  
        default=None,  
        description=("Indicates if the content is part of a table structure. "  
                     "This helps in applying appropriate table formatting in the Word document.")  
    )  
    notes: Optional[str] = Field(  
        default=None,  
        description=("Any additional notes or comments related to the paragraph. "  
                     "Useful for adding context or metadata that might not be part of the main content.")  
    )  
  
class TranslationsSchema(BaseModel):  
    """  
    Schema for a collection of translated paragraphs.  
  
    Attributes:  
        translations (List[TranslationsSchemaItem]): A list of translated paragraphs.   
                                                     Each item includes an optional header and the main content,  
                                                     both translated into English.  
    """  
    translations: List[TranslationsSchemaItem] = Field(  
        default_factory=list,  
        description=("A list of translated paragraphs. Each paragraph is represented by a TranslationsSchemaItem, "  
                     "which includes an optional header and the main content, both translated into English. "  
                     "This list captures the ordered sequence of translated text segments.")  
    )  


translate_template = """

You are a highly skilled translator. Please translate the following text from German to English with precision, ensuring that the translation is complete, clean, and retains the original meaning and tone.

Instructions:

Structure Preservation: Maintain the original structure by preserving the headers and their corresponding sections. Format headers to be bold and larger in size to ensure prominence when saved in a Word document. Headers in the text are denoted by one or more '#' symbols (e.g., #, ##, ###, ####, #####). Use the header_level field to capture the hierarchy of headers.
Tables:
If the original text contains tables or tabular data, translate the content while maintaining the table format.
Present tables in a plain text format, using clear column and row demarcations suitable for later conversion to a Word document. Use vertical bars (|) to separate columns and hyphens (-) to underline headers.
Example format:

| No. | Mandatory Requirements | Response Specification | Type |  
|-----|-------------------------|------------------------|------|  
| M1  | High technical and professional expertise... | At least 10 projects verifiable... | MUST |  
Use the table_format field to indicate if content is part of a table.
Lists and Bullet Points: If present, translate the content while maintaining list formatting. Use bullet points or numbered lists as appropriate. The list_type field should capture the type of list used.
Technical Terminology: Pay careful attention to technical terms to ensure accuracy and consistency with industry standards.
Clarity and Readability: Ensure that the translation reads naturally in English, making adjustments as necessary to maintain clarity and readability.
Additional Notes: Include any relevant notes or comments that might provide context to the translation. Use the notes field to capture this information.

Original Text:
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



