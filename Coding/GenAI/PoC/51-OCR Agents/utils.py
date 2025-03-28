"""
Utility functions for processing OCR results from Mistral API
"""
import json
from mistralai.models import OCRResponse


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