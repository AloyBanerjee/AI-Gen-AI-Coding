"""
Utility functions for processing OCR results from Mistral API
"""
import json
from pathlib import Path
from mistralai.models import OCRResponse
from mistral_ocr import MistralOCR
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, BinaryIO, List
import google.generativeai as genai
from phi.agent import Agent
from phi.model.groq import Groq


class Subsection(BaseModel):
    name: str = Field(..., description="Title of the subsection under a section.")
    text: str = Field(..., description="Text/content belonging to the subsection.")

class DocumentSectionExtraction(BaseModel):
    #document_name: str = Field(..., description="Name of the document from which the sections are extracted.")
    document_section_name: str = Field(..., description="Title of the section in the document.")
    document_subsection_name: Optional[List[Subsection]] = Field(
        None, description="List of subsections under the section, each with a title and associated text."
    )

class DocumentSectionExtractionList(BaseModel):
    sections: List[DocumentSectionExtraction]

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """
    Replace image placeholders in markdown with base64-encoded images.

    Args:
        markdown_str: Markdown text containing image placeholders
        images_dict: Dictionary mapping image IDs to base64 strings

    Returns:
        Markdown text with images replaced by base64 data
    """
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str


def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """
    Combine OCR text and images into a single markdown document.

    Args:
        ocr_response: Response from OCR processing containing text and images

    Returns:
        Combined markdown string with embedded images
    """
    markdowns: list[str] = []
    # Extract images from page
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        # Replace image placeholders with actual images
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    return "\n\n".join(markdowns)


def pretty_print_ocr(ocr_response: OCRResponse, max_chars: int = 1000) -> str:
    """
    Convert OCR response to a pretty-printed JSON string for display.
    
    Args:
        ocr_response: Response from OCR processing
        max_chars: Maximum number of characters to display

    Returns:
        Pretty-printed JSON string truncated to max_chars
    """
    response_dict = json.loads(ocr_response.model_dump_json())
    return json.dumps(response_dict, indent=4)[:max_chars]


def document_processing(uploaded_file, include_images: bool, show_raw_json: bool, api_key: str):
    # Initialize MistralOCR client
    ocr_client = MistralOCR(api_key=api_key)
    
    # Process the file based on its type
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension == '.pdf':
        ocr_response = ocr_client.process_pdf(
            file_content=uploaded_file, 
            file_name=uploaded_file.name,
            include_images=include_images
        )
    else:  # Image file
        ocr_response = ocr_client.process_image(
            file_content=uploaded_file,
            file_name=uploaded_file.name
        )
    
    return ocr_response


def generate_response(context, query, api_key, json_format=False):
    try:
       
        genai.configure(api_key=api_key)

        if len(context.pages) < 0:
            return "Error: No document content available to answer your question."

        prompt = f""" I have a document with the following content:
            {context}
            Based on this document, please answer the following question:
            {query}
            If you can find information related to the query in the document, please answer based on the information available in the context 
            If the document doesn't specifically mentione the exact information asked, please try to inform the same.
            """
        model = genai.GenerativeModel('gemini-1.5-flash') #gemma-3-27b-it

        generate_config = {
            "temparature":0.4,
            "top_p":0.8,
            "top_k":40,
            "max_output_tokens":2048,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
        ]
        if json_format:
            prompt = f"""{prompt}
            Please return the output in strict JSON format as specified.
            """
            response = model.generate_content(
                prompt,
                #generation_config = generate_config,
                #safety_settings = safety_settings
            )
        else:
            prompt = f"""{prompt}
            Please return the output in plain text format.
            """
            response = model.generate_content(
                prompt,                
                #generation_config = generate_config,
                #safety_settings = safety_settings
            )

        return response.text

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Error generating responses:{e}"

def docuemnt_section_subsecton_extraction(response: str) -> DocumentSectionExtractionList:    

    system_message = (
    "You are an agent that extracts structured information from documents. "
    "You must return your output in strict JSON format as specified."
    )
    user_message = """
        Please extract the sections and subsections from the following document. 
        Return the output in JSON format matching the following schema:

        {
        "sections": [
            {
            "document_section_name": "Section Title",
            "document_subsection_name": [
                {
                "name": "Subsection Title",
                "text": "Subsection content"
                }
            ]
            }
        ]
        }

        Document:
        <insert your document here>
        """
    
    ## Agent configuration 
    document_insight_generator = Agent(
        name="Structure Agent",
        model= Groq(id = 'llama-3.2-1b-preview'),
        description="Extract sections and subsections from the document. Return output in JSON format.",
        response_model=DocumentSectionExtractionList,
        structured_outputs=True,
        
    )

    struct_response = document_insight_generator.print_response(system_message + user_message + response)
    print("Structured Response:", struct_response)
    return struct_response.sections