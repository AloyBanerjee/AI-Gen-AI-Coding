import streamlit as st
import os
import tempfile
import gc
import base64
import time
from src.agentic_rag.tools.custom_tool import DocumentSearchTool
from dotenv import load_dotenv
from openai import OpenAI
from groundx import GroundX


load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROUNDX_API_KEY = os.getenv("GROUNDX_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ===========================
#   Streamlit Setup
# ===========================
st.set_page_config(page_title='Complex RAG', page_icon='🕷️', layout='wide')
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None  # Store the DocumentSearchTool

def delete_docums(bucket_id):
    groundx = GroundX(api_key=GROUNDX_API_KEY)
    response = groundx.buckets.delete(
        bucket_id=bucket_id,
    )
    return response

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the uploaded PDF in an iframe."""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)

def fetch_answer(prompt: str, bucket_id=None, process_id=None):  
    groundx = GroundX(api_key=GROUNDX_API_KEY)

    client = OpenAI(api_key=os.environ.get(OPENAI_API_KEY))  

    with st.status("Checking document processing status...", expanded=True) as status_box:
        while True:
            status_response = groundx.documents.get_processing_status_by_id(process_id=process_id)
            status = status_response.ingest.status

            if status == 'complete':
                status_box.update(label="✅ Document processing complete.", state="complete")
                
                completion_model = "gpt-4o"
                instruction = '''You are a helpful virtual assistant that answers questions using the content below.
                                Your task is to create detailed answers to the questions by combining your understanding of the world with the content provided below.
                                Do not share links.'''

                content_response = groundx.search.content(id=bucket_id, query=prompt)
                results = content_response.search
                llm_text = results.text                
                completion = client.chat.completions.create(
                    model=completion_model,
                    messages=[
                        {
                            "role": "system",
                            "content": """%s
                        ===
                        %s
                        ===
                        """ % (instruction, llm_text),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                
                return prompt, results.score, completion.choices[0].message.content, llm_text  # Return results
           
            elif status == 'error':
                status_box.update(label="❌ An error occurred during document processing.", state="error")
                return None, None, None, None  # Handle error case

            else:
                status_box.update(label=f"⏳ Current status: {status}. Checking again in 3 seconds...")
                time.sleep(3)  # Wait before checking again

# ===========================
#   Sidebar
# ===========================
with st.sidebar:
    st.header("Add Your PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # If there's a new file and we haven't set pdf_tool yet...
        if st.session_state.pdf_tool is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                with st.spinner("🔄 Indexing PDF... Please wait..."):
                    # st.session_state.pdf_tool = DocumentSearchTool(file_path="/Users/akshay/Eigen/ai-engineering-hub/agentic_rag_deepseek/knowledge/dspy.pdf")
                    st.session_state.pdf_tool = DocumentSearchTool(file_path=temp_file_path)

            st.success("PDF indexed! Ready to chat.")

        # Optionally display the PDF in the sidebar
        display_pdf(uploaded_file.getvalue(), uploaded_file.name)

    st.button("Clear Chat", on_click=reset_chat)
    if st.session_state.pdf_tool != None:
        if st.session_state.pdf_tool.bucket_id:
            st.button("Delete Document", on_click=delete_docums, args=[st.session_state.pdf_tool.bucket_id])



# ===========================
#   Main Chat Interface
# ===========================
st.markdown('''
    # Complex Document RAG ''' , unsafe_allow_html=True)

# Render existing conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question about your PDF...")

if prompt:
    # 1. Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Run RAG on the prompt
    query, score, result, llm_results = fetch_answer(prompt, st.session_state.pdf_tool.bucket_id, st.session_state.pdf_tool.process_id)
    with st.spinner("Thinking..."):
        time.sleep(1)
    

    # 3. Save assistant's message to session
    st.session_state.messages.append({"role": "assistant", "content": result})

    # 4. Show the final response

    tab1, tab2 = st.tabs(["💬 Responce", "📊 Metadata"])
    
    with tab1:
        with st.expander("Thinking...",expanded=False, icon=None):  
            st.subheader("🤔 Thinking...")   
            st.info(llm_results)      
        st.subheader("🔍 Responce") 
        message_placeholder = st.empty()
        message_placeholder.markdown(result)

    with tab2:
        st.subheader("🌐 Response Metadata")
        if prompt:
            st.markdown(f"**Query:** {query}")
            st.markdown(f"**Score:** {score:.4f}")  # Display score with 4 decimal places
            st.markdown("**Processed using GroundX RAG Model**")    
 

