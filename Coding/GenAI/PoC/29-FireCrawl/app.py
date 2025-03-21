import streamlit as st
import os
import gc
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import time
import pandas as pd
from typing import Dict, Any
import base64
from pydantic import BaseModel, Field
import inspect


st.set_page_config(page_title='FireCarwl', page_icon='🕷️', layout='wide')
load_dotenv()
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")



# CSS for a blue-themed table

overall_css = """
    <style>

    /* Customize file uploader     
    [data-testid="stFileUploader"] > label {
        background-color: #e8f4ff; 
        color: #004085;           
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #80bfff; 
        cursor: pointer;
    }*/
    [data-testid="stFileUploader"] button {
        
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #0056b3; /* Darker blue on hover */
        color: white;
        border-color: #0056b3; /* Darker blue border on hover */
    }


    [data-testid="stFileUploader"] > div {
        background-color: #f2f9ff; /* Very light blue background for file area */
        border: 1px solid #80bfff; /* Light blue border */
        border-radius: 5px;
    }

    /* Customize select box */
    [data-baseweb="select"] > div {
        background-color: #e8f4ff;  /* Light blue background */
        color: #004085;             /* Dark blue text */
        font-weight: bold;
        border: 1px solid #80bfff;  /* Light blue border */
        border-radius: 5px;
    }

    [data-baseweb="select"] .css-1hb7zxy-Input {
        color: #004085;             /* Dark blue text inside the dropdown */
    }

    [data-baseweb="select"] [role="option"] {
        background-color: #f2f9ff;  /* Very light blue for dropdown options */
        color: #004085;             /* Dark blue text */
    }

    [data-baseweb="select"] [role="option"]:hover {
        background-color: #cce7ff;  /* Slightly darker blue on hover */
    }
    /* Customize Headers */
    h1, h2, h3, h4 {
        font-family: 'Arial', sans-serif;
        color: #004085;
        font-weight: bold;
        text-align: left;
        padding: 10px;
    }

    h1 {
        font-size: 36px;
        background: linear-gradient(to right, #007bff, #80bfff);
        -webkit-background-clip: text;
        color: transparent;
    }

    h2 {
        font-size: 28px;
        #border-bottom: 3px solid #007bff;
        padding-bottom: 5px;
    }

    h3 {
        font-size: 24px;
        #border-bottom: 2px solid #007bff;
    }

    h4 {
        font-size: 20px;
        #border-bottom: 1px solid #007bff;
    }

    /* Customize Text Area */
    textarea {
        background-color: #f2f9ff;
        color: #004085;
        font-size: 16px;
        font-weight: bold;
        border: 1px solid #80bfff;
        border-radius: 5px;
        padding: 10px;
    }

    textarea:focus {
        border-color: #007bff;
        outline: none;
        box-shadow: 0px 0px 5px #007bff;
    }

    
    /* Customize Text Input Box */
    input[type="text"] {
        background-color: #f2f9ff;
        color: #004085;
        font-size: 16px;
        font-weight: bold;
        border: 1px solid #80bfff;
        border-radius: 5px;
        padding: 8px;
        width: 100%;
    }

    input[type="text"]:focus {
        border-color: #007bff;
        outline: none;
        box-shadow: 0px 0px 5px #007bff;
    }

    /* Customize Labels */
    label {
        font-size: 16px;
        font-weight: bold;
        color: #004085;
    }
    </style>
"""
btncss = """
    <style>
    div.stButton > button {
        background-color: #004085; /* Blue background */
        color: white; /* White text */
        border: 2px solid #004085; /* Blue border */
        padding: 0.5em 1em; /* Padding inside the button */
        border-radius: 4px; /* Rounded corners */
        font-size: 16px; /* Font size */
        font-weight: bold; /* Bold text */
        cursor: pointer; /* Pointer cursor on hover */
    }
    div.stButton > button:hover {
        background-color: #0056b3; /* Darker blue on hover */
        color: white;
    }
    </style>
    """


st.markdown(btncss,
    unsafe_allow_html=True
)
st.markdown(overall_css,
    unsafe_allow_html=True
)




@st.cache_resource
def load_app():
    app = FirecrawlApp(api_key=firecrawl_api_key)
    return app

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "schema_fields" not in st.session_state:
    st.session_state.schema_fields = [{"name": "", "type": "str"}]

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def create_dynamic_model(fields):
    """Create a dynamic Pydantic model from schema fields."""
    field_annotations = {}
    for field in fields:
        if field["name"]:
            # Convert string type names to actual types
            type_mapping = {
                "str": str,
                "bool": bool,
                "int": int,
                "float": float
            }
            field_annotations[field["name"]] = type_mapping[field["type"]]
    
    # Dynamically create the model class
    return type(
        "ExtractSchema",
        (BaseModel,),
        {
            "__annotations__": field_annotations
        }
    )

def create_schema_from_fields(fields):
    """Create schema using Pydantic model."""
    if not any(field["name"] for field in fields):
        return None
    
    model_class = create_dynamic_model(fields)
    return model_class.model_json_schema()

def convert_to_table(data):
    """Convert a list of dictionaries to a markdown table."""
    if not data:
        return ""
    
    # Convert only the data field to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert DataFrame to markdown table
    return df.to_markdown(index=False)

def dict_to_dataframe(data):
    """
    Converts a dictionary into a Pandas DataFrame where:
    - Dictionary keys become column names
    - Values become rows
    """
    df = pd.DataFrame([data])  # Convert dictionary to DataFrame (single-row DataFrame)
    return df.to_markdown(index=False)  # Convert to markdown for better readability


def stream_text(text: str, delay: float = 0.001) -> None:
    """Stream text with a typing effect."""
    placeholder = st.empty()
    displayed_text = ""
    
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(delay)
    
    return placeholder

# Main app layout
st.markdown("""
    # 🕷️ Website Crawling using <img src="data:image/png;base64,{}" style="vertical-align: top; margin: 0px 5px" width="100px">
""".format(base64.b64encode(open("assets/firecrawl.png", "rb").read()).decode()), unsafe_allow_html=True)


tab1, tab2 = st.tabs(["⚙️ Configuration", "💬 Chat Window"])
with tab1:
    st.header('⚙️ Configuration')
    website_url = st.text_input("Enter Website URL", placeholder="https://example.com")
    if website_url:
        st.write('Website URL:', website_url)

    st.subheader("Schema Builder (Optional)")
    for i, field in enumerate(st.session_state.schema_fields):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            field["name"] = st.text_input(
                "Field Name",
                value=field["name"],
                key=f"name_{i}",
                placeholder="e.g., company_mission"
            )
        
        with col2:
            field["type"] = st.selectbox(
                "Type",
                options=["str", "bool", "int", "float"],
                key=f"type_{i}",
                index=0 if field["type"] == "str" else ["str", "bool", "int", "float"].index(field["type"])
            )
    if st.session_state.schema_fields:    
        st.write('Schema Fields:', st.session_state.schema_fields)
        if len(st.session_state.schema_fields) < 5:  # Limit to 5 fields
            if st.button("Add Field ➕"):
                st.session_state.schema_fields.append({"name": "", "type": "str"})

with tab2:
    st.header('💬 Chat Window')
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the website..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if not website_url:
                st.error("Please enter a website URL first!")
            else:
                try:
                    with st.spinner("Extracting data from website..."):
                        #st.write('Loading the APP')
                        app = load_app()
                        #st.write('APP is loaded')
                        schema = create_schema_from_fields(st.session_state.schema_fields)
                        #st.write(schema)
                        extract_params = {
                            'prompt': prompt
                        }
                        if schema:
                            extract_params['schema'] = schema
                            
                        data = app.extract(
                            [website_url],
                            extract_params
                        )
                        st.markdown(data['data'])
                        # check if data['data'] is a list, if yes, pass data['data'] to convert_to_table
                        #st.write((data['data'].keys()))
                        if isinstance(data['data'], list):
                            table = convert_to_table(data['data'])
                        else:
                            # find the first key in data['data']
                            # key = list(data['data'].keys())
                            # table = convert_to_table(data['data'])
                            table = dict_to_dataframe(data['data'])
                        
                        placeholder = stream_text(table)
                        st.session_state.messages.append({"role": "assistant", "content": table})
                        # st.markdown(table)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Footer
# Footer with HTML and CSS
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #004085; /* Change this value for a new background color */
        color: white;
        text-align: center;
        padding: 10px 0;
        font-family: Arial, sans-serif;
        font-size: 14px;
        border-top: 2px solid #007bff; /* Optional border color */
        z-index: 1000;
        height: 40px;
    }

    .footer a {
        color: #ffdd00; /* Link color */
        text-decoration: none;
        margin: 0 10px;
    }

    .footer a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="footer">
        <p>
            Copyright &copy; 2025 <strong>GEN AI Enthusiats</strong> |
            <a href="https://haha.com" target="_blank">Visit Our Website</a> |
            <a href="mailto:support@example.com">Contact Us</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)