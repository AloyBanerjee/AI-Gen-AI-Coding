"""
Streamlit app for OCR using Mistral AI
"""
import os
import tempfile
import streamlit as st
from pathlib import Path

from mistral_ocr import MistralOCR
from utils import get_combined_markdown, pretty_print_ocr

import google.generativeai as genai
#from google import genai
#from google.genai import types

from dotenv import load_dotenv

load_dotenv()

global google_api_key
google_api_key = os.getenv('GOOGLE_API_KEY')


def generate_response(context, query):
    try:
       
        genai.configure(api_key=google_api_key)

        if len(context.pages) < 0:
            return "Error: No document content available to answer your question."

        prompt = f""" I have a document with the following content:
            {context}
            Based on this document, please answer the following question:
            {query}
            If you can find information related to the query in the document, please answer based on the information available in the context 
            If the document doesn't specifically mentione the exact information asked, please try to inform the same.
            """
        #client = genai.Client(api_key=google_api_key)

        model = genai.GenerativeModel('gemini-1.5-flash') #gemma-3-27b-it


        # response = client.models.generate_content(
        #     model="gemini-2.0-flash",
        #     contents=['Do these look store-bought or homemade?', img],
        #     config=types.GenerateContentConfig(
        #     safety_settings=[
        #         types.SafetySetting(
        #             category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        #             threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        #         ),
        #     ]
        #     )
        # )


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


# Set page configuration
st.set_page_config(
    page_title="Mistral OCR App",
    page_icon="📄",
    layout="wide"
)
# Initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "api_key_saved" not in st.session_state:
    st.session_state.api_key_saved = False
if "document_content" not in st.session_state:
    st.session_state.document_content = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    """Main function for the Streamlit app."""
    
    col1, col2 = st.columns([3,1])
    with col1:
        # Header
        st.title("📜 DeepScan OCR - Powered by Mistral")
    with col2:
        st.image(r'H:\Interview Preparation\Coding\GenAI\Tryouts\51-OCR Agents\Mistral-AI-logo.jpg')
    st.info("""
    This application leverages Mistral AI's advanced Optical Character Recognition (OCR) technology to accurately extract both text and images from a variety of file formats, including PDFs and image files. Simply upload your documents below, and the system will process them to retrieve valuable content efficiently. 
    Whether you need to extract printed or handwritten text, this tool ensures high precision and ease of use.
    """)
    tabs = st.tabs([
        "🔧 Configuration",  
        "🏗️ Information Extraction for Chat",
        "🌟💬 Chat with Extracted Details"
    ])

    # Sidebar for API key
    with tabs[0]:
        st.header("API Configuration")
        # api_key = st.text_input("Enter your Mistral API key", type="password")
        # if api_key:
        #     st.session_state.api_key = api_key
        #     st.success('API Key is configure sucessfully')
        # Disable input if key is saved
        disabled = st.session_state.api_key_saved

        api_key = st.text_input("Enter your Mistral API key", type="password", disabled=disabled)

        if api_key and not st.session_state.api_key_saved:
            st.session_state.api_key = api_key
            st.session_state.api_key_saved = True
            st.success("API Key is configured successfully")

        # Optionally, allow users to re-enter a key
        if st.session_state.api_key_saved:
            if st.button("Reset API Key"):
                st.session_state.api_key = ""
                st.session_state.api_key_saved = False
                st.rerun()

        st.markdown("### Options")
        include_images = st.checkbox("Include images in results", value=True)
        show_raw_json = st.checkbox("Show raw JSON response", value=False)
        st.markdown("""
        ### How to Obtain a Mistral API Key:  
        1. Visit the [Mistral AI Platform](https://console.mistral.ai/)  
        2. Sign up or log in to your account  
        3. Go to the **API** section  
        4. Generate a new API key
                """)
        

    with tabs[1]:
        # Main content
        uploaded_file = st.file_uploader("Upload a PDF or image file", type=["pdf", "png", "jpg", "jpeg"])
        
        if uploaded_file is not None and st.session_state.api_key :
            with st.spinner("Processing file..."):
                try:
                    # Initialize MistralOCR client
                    ocr_client = MistralOCR(api_key=st.session_state.api_key )
                    
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
                    
                    # Display the OCR results
                    st.success("File processed successfully!")
                    
                    # Display raw JSON if requested
                    if show_raw_json:
                        with st.expander("Raw JSON Response"):
                            st.code(pretty_print_ocr(ocr_response), language="json")
                    
                    # Display the combined markdown with text and images
                    st.header("OCR Results")
                    st.session_state.document_content =  ocr_response
                    st.markdown(get_combined_markdown(ocr_response), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        elif uploaded_file is not None and not st.session_state.api_key:
            st.warning("Please enter your Mistral API key in the sidebar.")       

    with tabs[2]:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        # Input for user query
        if prompt := st.chat_input("Ask a question about your document..."):
            # Check if Google API key is available
            if not google_api_key:
                st.error("Google API key is required for generating responses. Please configure it.")
            else:
                st.session_state.messages.append({"role":"user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    document_content = st.session_state.document_content
                    st.sidebar.write(len(document_content.pages))
                    st.sidebar.write(document_content.pages)
                    response = generate_response(document_content,prompt)
                    st.markdown(response)        
                    st.session_state.messages.append({"role":"assistant", "content": response})


if __name__ == "__main__":
    main()