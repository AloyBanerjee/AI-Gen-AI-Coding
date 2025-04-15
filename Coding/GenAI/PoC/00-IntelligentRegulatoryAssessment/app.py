import os
import pandas as pd
import json
import tempfile
import streamlit as st
from pathlib import Path
from mistral_ocr import MistralOCR
from utils import get_combined_markdown, pretty_print_ocr, document_processing, docuemnt_section_subsecton_extraction, generate_response
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

global google_api_key
#google_api_key = os.getenv('GOOGLE_API_KEY')

## COMMON FUNCTION -- START##

def display_ocr_response(ocr_response, show_raw_json: bool) -> str:
   # Display raw JSON if requested
    if show_raw_json:
        with st.expander("Raw JSON Response"):
            st.code(pretty_print_ocr(ocr_response), language="json")
    
    # Display the combined markdown with text and images
    st.header("OCR Results")
    st.session_state.document_content =  ocr_response
    st.markdown(get_combined_markdown(ocr_response), unsafe_allow_html=True)

## COMMON FUNCTION -- END##

# Set page configuration
st.set_page_config(
    page_title="Mistral OCR App",
    page_icon="🛠️",
    layout="wide"
)
# Initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""
if "api_key_saved" not in st.session_state:
    st.session_state.api_key_saved = False
if "old_document_content" not in st.session_state:
    st.session_state.old_document_content = ""
if "new_document_content" not in st.session_state:
    st.session_state.new_document_content = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    """Main function for the Streamlit app."""
    
    col1, col2 = st.columns([3,1])
    with col1:
        # Header
        st.markdown(
            """
            <h1>📜 RegInsight <sup style="font-size: 14px; color: green;">Intelligent Regulatory Assessment</sup></h1>
            """,
            unsafe_allow_html=True
        )
    with col2:
        logo_col, logo_col2 = st.columns([1, 1])
        with logo_col:
            st.image(r'H:\Interview Preparation\Coding\GenAI\Tryouts\00-IntelligentRegulatoryAssessment\Gemini-Logo.png', width=200) 
        with logo_col2:
            st.image(r'H:\Interview Preparation\Coding\GenAI\Tryouts\00-IntelligentRegulatoryAssessment\Mistral-AI-logo.jpg')
    st.info("""
    This application utilizes advanced AI-driven capabilities to intelligently extract and assess regulatory content from various document formats, including PDFs and images.
            By simply uploading your files below, the system accurately identifies and retrieves both text and visual elements, supporting detailed regulatory review and compliance checks. Whether the content is printed or handwritten, this tool ensures high precision, relevance, and ease of use for efficient regulatory assessment.
            In addition to content extraction, the system enables side-by-side comparison of different versions of regulatory documents, highlighting key changes and generating actionable insights. This ensures greater clarity, compliance accuracy, and efficiency in your regulatory review process.
    """)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔧 Configuration",         
        "📤 Regulatory Document Upload", 
        "🏗️ Information Extraction from Uploaded Document",
        "💬 Chat with Extracted Details",
        "🧲 Section & Subsection Extraction",
        "🆚 Document Comparison",
    ])

    
    with tab1:

        head_col1, head_col2 = st.columns([5, 1])
        with head_col1:
            st.header(f""":rainbow[API Configuration]""")
        with head_col2:
            default_config = st.button('Default API KEYs', type="primary")
            if default_config:
                st.session_state.api_key = os.getenv('MISTRAL_API_KEY')
                st.session_state.google_api_key = os.getenv('GOOGLE_API_KEY')
                st.session_state.api_key_saved = True                
                # Disable input if key is saved
                disabled = st.session_state.api_key_saved
        if default_config:
            st.success("Default API Key is configured successfully")

        
        # Disable input if key is saved
        disabled = st.session_state.api_key_saved

        api_key = st.text_input("Enter your Mistral API key", type="password", disabled=disabled)
        google_api_key = st.text_input("Enter your Google API key", type="password", disabled=disabled)

        if api_key and google_api_key and not st.session_state.api_key_saved:
            st.session_state.api_key = api_key
            st.session_state.google_api_key = google_api_key
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

        st.markdown("### API Key Instructions")
        st.markdown("""
                To use this application, you need to obtain an API key from Mistral and Google Generative AI. Follow the instructions below to get your keys:
                """)
        api_key_inst_col1, api_key_inst_col2 = st.columns([1, 3])
        with api_key_inst_col1:
            st.markdown("""
                    ### How to Obtain a Mistral API Key:  
                    1. Visit the [Mistral AI Platform](https://console.mistral.ai/)  
                    2. Sign up or log in to your account  
                    3. Go to the **API** section  
                    4. Generate a new API key
                    """)
        with api_key_inst_col2:
            st.markdown("""
                ### How to Obtain a Google Generative AI API Key:
                1. Visit the [Google AI Studio](https://makersuite.google.com/)
                2. Sign in with your Google account
                3. Navigate to the **API Keys** section
                4. Click **Create API Key** to generate a new key
                5. Copy and securely store your API key for use in applications
                """)


    with tab2:
        # Main content
        #uploaded_file = st.file_uploader("Upload a PDF or image file", type=["pdf", "png", "jpg", "jpeg"])
        
        st.header(f""":rainbow[Document Upload]""")
        with st.expander("💡 :rainbow[Regulatory Document Selection & Upload]", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                # the first file upload field, the specific ui element that allows you to upload file 1
                File1 = st.file_uploader('Upload File 1 (Document A) - **New Version**', type=["pdf", "png", "jpg", "jpeg"], key="doc_1")
            with col2:
                # the second file upload field, the specific ui element that allows you to upload file 2
                File2 = st.file_uploader('Upload File 2 (Document B) - **Old Version**', type=["pdf", "png", "jpg", "jpeg"], key="doc_2")
                # when both files are uploaded it saves the files to the directory, creates a path, and invokes the

            upload_proccess = st.button("Upload and Process Files", type="primary", disabled=not (File1 and File2))
            if upload_proccess and File1 is not None and File2 is not None and st.session_state.api_key and st.session_state.google_api_key:
                with st.spinner("Processing file..."):
                    try:
                        # Process the files using Mistral OCR
                        ocr_response_file1=document_processing(File1, include_images, show_raw_json, st.session_state.api_key)
                        ocr_response_file2=document_processing(File2, include_images, show_raw_json, st.session_state.api_key)
                        
                        # Save the processed files responce to session object
                        st.session_state.old_document_content = ocr_response_file2
                        st.session_state.new_document_content = ocr_response_file1

                        # Display the OCR results
                        st.success("File processed successfully!")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")


    with tab3:
        if st.session_state.new_document_content is not None and st.session_state.old_document_content is not None and st.session_state.api_key and st.session_state.google_api_key:                
            try: 
                tabs = st.tabs([
                        "📑 :rainbow[OCR Results for Document A (New Version)]",         
                        "📑 :rainbow[OCR Results for Document B (Old Version)]",                         
                    ])
                with tabs[0]:                    
                    display_ocr_response(st.session_state.new_document_content, show_raw_json)
                    
                with tabs[1]:
                    display_ocr_response(st.session_state.old_document_content, show_raw_json)                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")        
        else:
            st.warning("Please upload both the files and enter your API keys before proceeding further.")     


    with tab4:
        if st.session_state.new_document_content is not None and st.session_state.old_document_content is not None and st.session_state.api_key and st.session_state.google_api_key:                
            
            colchat1, colchat2 = st.columns([3, 1])
            with colchat1:
                st.write("## :rainbow[Chat with Extracted Details]")
                st.info("Chat with the extracted details from the document. Ask questions related to the content.")
            with colchat2:
                st.button("Clear Chat", type="primary", on_click=lambda: st.session_state.messages.clear())

            
            document_list = ["(Document A) - **New Version**", "(Document B) - **Old Version**"]

            st.markdown("### 💬 Select a Document to Chat With")
            selected_doc = st.selectbox("Choose a document:", document_list, key = "doc_chat")

            if selected_doc:
                st.info(f"You selected: **{selected_doc}**")
                if selected_doc == "(Document A) - **New Version**":
                    document_content = st.session_state.new_document_content    
                elif selected_doc == "(Document B) - **Old Version**":
                    document_content = st.session_state.old_document_content    

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Input for user query
            if prompt := st.chat_input("Ask a question about your document..."):
                # Check if Google API key is available
                if not st.session_state.google_api_key:
                    st.error("Google API key is required for generating responses. Please configure it.")
                else:
                    st.session_state.messages.append({"role":"user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        document_content = st.session_state.document_content
                        with st.expander("Document Content", expanded=False):
                            st.subheader("Document Content:")
                            if len(document_content.pages) > 0:
                                st.write("Number of pages in the document:")
                            st.info(len(document_content.pages))
                            st.write(document_content.pages)
                        response = generate_response(document_content,prompt, st.session_state.google_api_key, json_format=False)
                        st.markdown(response)        
                        st.session_state.messages.append({"role":"assistant", "content": response})
        else:
            st.warning("Please upload both the files and enter your API keys before proceeding further.")   


    with tab5:
        if st.session_state.new_document_content is not None and st.session_state.old_document_content is not None and st.session_state.api_key and st.session_state.google_api_key:                
            
            colchat1, colchat2 = st.columns([3, 1])
            with colchat1:
                st.write("## :rainbow[Section Extraction and Comparison]")
                st.info("Section Extraction and Comparison of the documents.")
            st.markdown("### 🆚 Select a Document to Compare")
           
           
            document_list = ["(Document A) - **New Version**", "(Document B) - **Old Version**"]

            st.markdown("### 🧲 Select a Document for section extraction")
            selected_doc = st.selectbox("Choose a document:", document_list, key="doc_compare")

            if selected_doc:
                st.info(f"You selected: **{selected_doc}**")
                if selected_doc == "(Document A) - **New Version**":
                    document_content = st.session_state.new_document_content    
                elif selected_doc == "(Document B) - **Old Version**":
                    document_content = st.session_state.old_document_content 


            extract_section = st.button("Extract Section", type="primary")
            if extract_section:
                with st.spinner("Extracting sections..."):
                    try:                        
                        prompt = 'what are the different section and subsection available in the document and extract the text contains in each subsection and show in tree view manner'
                        response = generate_response(document_content, prompt, st.session_state.google_api_key, json_format=False)
                        st.markdown(response)
                        json_response = generate_response(document_content, prompt, st.session_state.google_api_key, json_format=True)
                        struct_output = docuemnt_section_subsecton_extraction(json_response)

                        st.write("### Extracted Sections and Subsections")
                        st.write("Document Name:", selected_doc)
                        st.write("Sections and Subsections:")
                        for section in struct_output:
                            st.write("Section Name:", section.document_section_name)
                            if section.document_subsection_name:
                                for subsection in section.document_subsection_name:
                                    st.write("Subsection Name:", subsection.name)
                                    st.write("Text:", subsection.text)
                            else:
                                st.write("Subsection Name: N/A")
                                st.write("Text: N/A")
                        
                        #extracted_sections = docuemnt_section_subsecton_extraction(response, st.session_state.api_key)
                        # st.write(extracted_sections)

                        # # Flatten into rows for table
                        # table_data = []
                        # for section in extracted_sections:
                        #     section_name = section.document_section_name
                        #     doc_name = selected_doc or "N/A"
                        #     if section.document_subsection_name:
                        #         for subsection in section.document_subsection_name:
                        #             table_data.append({
                        #                 "Document Name": doc_name,
                        #                 "Section": section_name,
                        #                 "Subsection": subsection.name,
                        #                 "Text": subsection.text
                        #             })
                        #     else:
                        #         table_data.append({
                        #             "Document Name": doc_name,
                        #             "Section": section_name,
                        #             "Subsection": "N/A",
                        #             "Text": "N/A"
                        #         })

                        # # Create and display DataFrame
                        # df = pd.DataFrame(table_data)
                        # st.write("### Extracted Sections and Subsections")
                        # st.dataframe(df)

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please upload both the files and enter your API keys before proceeding further.")  

if __name__ == "__main__":
    main()