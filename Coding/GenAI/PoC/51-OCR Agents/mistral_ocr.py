"""
Module for interacting with Mistral's OCR API.
"""
from pathlib import Path
import tempfile
from typing import Optional, BinaryIO

from mistralai import Mistral
from mistralai.models import OCRResponse
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk


class MistralOCR:
    """
    A class to handle OCR using Mistral's API.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Mistral client with the provided API key.
        
        Args:
            api_key: Mistral API key
        """
        self.api_key = api_key
        self.client = Mistral(api_key=api_key)
    
    def process_pdf(self, file_content: BinaryIO, file_name: str, include_images: bool = True) -> OCRResponse:
        """
        Process a PDF file using Mistral's OCR service.
        
        Args:
            file_content: The content of the PDF file as bytes
            file_name: The name of the file
            include_images: Whether to include images in the response
            
        Returns:
            OCR response with text and optionally images
        """
        # Upload PDF file to Mistral's OCR service
        uploaded_file = self.client.files.upload(
            file={
                "file_name": file_name,
                "content": file_content.read(),
            },
            purpose="ocr",
        )
        
        # Get URL for the uploaded file
        signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        
        # Process PDF with OCR
        pdf_response = self.client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=include_images
        )
        
        return pdf_response
    
    def process_image(self, file_content: BinaryIO, file_name: str) -> OCRResponse:
        """
        Process an image file using Mistral's OCR service.
        
        Args:
            file_content: The content of the image file as bytes
            file_name: The name of the file
            
        Returns:
            OCR response with text
        """
        # Upload image file to Mistral's OCR service
        uploaded_file = self.client.files.upload(
            file={
                "file_name": file_name,
                "content": file_content.read(),
            },
            purpose="ocr",
        )
        
        # Get URL for the uploaded file
        signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        
        # Process image with OCR
        image_response = self.client.ocr.process(
            document=ImageURLChunk(image_url=signed_url.url),
            model="mistral-ocr-latest",
        )
        
        return image_response