import streamlit as st
from pathlib import Path
import os
from doc_comparer import doc_compare
from dotenv import load_dotenv
import streamlit as st
import json
from typing import Iterable
from moa import *
from streamlit_ace import st_ace
import copy
from io import StringIO
import re
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# load environment variables
load_dotenv()

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = OpenAIEmbeddings()
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if  "doc_changes" not in st.session_state:
    st.session_state.doc_changes = None

## Common Functionalities
def set_moa_agent(
    main_model: str = default_config['main_model'],
    cycles: int = default_config['cycles'],
    layer_agent_config: dict[dict[str, any]] = copy.deepcopy(layer_agent_config_def),
    main_model_temperature: float = 0.1,
    override: bool = False
):
    if override or ("main_model" not in st.session_state):
        st.session_state.main_model = main_model
    else:
        if "main_model" not in st.session_state: st.session_state.main_model = main_model 

    if override or ("cycles" not in st.session_state):
        st.session_state.cycles = cycles
    else:
        if "cycles" not in st.session_state: st.session_state.cycles = cycles

    if override or ("layer_agent_config" not in st.session_state):
        st.session_state.layer_agent_config = layer_agent_config
    else:
        if "layer_agent_config" not in st.session_state: st.session_state.layer_agent_config = layer_agent_config

    if override or ("main_temp" not in st.session_state):
        st.session_state.main_temp = main_model_temperature
    else:
        if "main_temp" not in st.session_state: st.session_state.main_temp = main_model_temperature

    cls_ly_conf = copy.deepcopy(st.session_state.layer_agent_config)
    
    if override or ("moa_agent" not in st.session_state):
        st.session_state.moa_agent = from_config(
            main_model=st.session_state.main_model,
            cycles=st.session_state.cycles,
            layer_agent_config=cls_ly_conf,
            temperature=st.session_state.main_temp
        )

    del cls_ly_conf
    del layer_agent_config

def stream_response(messages: Iterable[ResponseChunk]):
    layer_outputs = {}
    for message in messages:
        if message['response_type'] == 'intermediate':
            layer = message['metadata']['layer']
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append(message['delta'])
        else:
            # Display accumulated layer outputs
            for layer, outputs in layer_outputs.items():
                st.write(f"Layer {layer}")
                cols = st.columns(len(outputs))
                for i, output in enumerate(outputs):
                    with cols[i]:
                        st.expander(label=f"Agent {i+1}", expanded=False).write(output)
            
            # Clear layer outputs for the next iteration
            layer_outputs = {}
            
            # Yield the main agent's output
            yield message['delta']

def capture_stream(generator):
    """Captures streamed output from `st.write_stream()` and returns full text."""
    output_buffer = StringIO()  # Temporary buffer to store streamed content

    def write_and_capture(text):
        output_buffer.write(text + "\n")  # Store the streamed content
        return text  # Also return it to Streamlit

    st.write_stream((write_and_capture(chunk) for chunk in generator))  # Capture stream
    # for chunk in generator:
    #   write_and_capture(chunk) 
        
    return output_buffer.getvalue().strip()  # Return full captured output after completion

def process_stream(full_text):
    """Processes full text, extracting <think> content separately."""
    # Extract <think> content
    think_contents = re.findall(r"<think>(.*?)</think>", full_text, re.DOTALL)

    # Remove <think> tags from the main text
    processed_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()

    return processed_text.strip(), [t.strip() for t in think_contents]  # Trim spaces

def format_reasoning_response(thinking_content):
    """Format assistant content by removing think tags."""
    return (
        thinking_content.replace("<think>\n\n</think>", "")
        .replace("<think>", "")
        .replace("</think>", "")
    )

def process_thinking_phase(stream):
    """Process the thinking phase of the assistant's response."""
    thinking_content = ""
    with st.status("Thinking...", expanded=True) as status:
        think_placeholder = st.empty()
        
        for chunk in stream:
            content = chunk["message"]["content"] or ""
            thinking_content += content
            
            if "<think>" in content:
                continue
            if "</think>" in content:
                content = content.replace("</think>", "")
                status.update(label="Thinking complete!", state="complete", expanded=False)
                break
            think_placeholder.markdown(format_reasoning_response(thinking_content))
    return thinking_content

def process_response_phase(stream):
    """Process the response phase of the assistant's response."""
    response_placeholder = st.empty()
    response_content = ""
    for chunk in stream:
        content = chunk["message"]["content"] or ""
        response_content += content
        response_placeholder.markdown(response_content)
    return response_content

st.set_page_config(
    page_title="Long Document Comparison",
    page_icon=r'H:\Interview Preparation\Coding\GenAI\Tryouts\41-MixtureofAgent-Notebook\static\favicon.ico',
        menu_items={
        'About': "## Groq Mixture-Of-Agents \n Powered by [Groq](https://groq.com)"
    },
    layout="wide"
)

set_moa_agent()
   


# title of the streamlit app
st.title(f""":rainbow[Long Document Comparison]""")

tab1, tab2 = st.tabs([
    "🛠️ MOA Configuration", 
    "📄 Document Comparison"   
])

with tab1:
    # config_form = st.form("Agent Configuration", border=False)
    st.title(f""":rainbow[MOA Configuration]""")
    with st.form("Agent Configuration", border=False):
        if st.form_submit_button("Use Recommended Config"):
            try:
                set_moa_agent(
                    main_model=rec_config['main_model'],
                    cycles=rec_config['cycles'],
                    layer_agent_config=layer_agent_config_rec,
                    override=True
                )
                st.session_state.layer_agent_config = layer_agent_config_rec
                st.session_state.messages = []
                st.success("Configuration updated successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")
        # Main model selection
        new_main_model = st.selectbox(
            "Select Main Model",
            options=valid_main_model_list,
            index=valid_main_model_list.index(st.session_state.main_model)
        )

        # Cycles input
        new_cycles = st.number_input(
            "Number of Layers",
            min_value=1,
            max_value=10,
            value=st.session_state.cycles
        )

        # Main Model Temperature
        main_temperature = st.number_input(
            label="Main Model Temperature",
            value=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.1
        )

        # Layer agent configuration
        tooltip = "Agents in the layer agent configuration run in parallel _per cycle_. Each layer agent supports all initialization parameters of [Langchain's ChatGroq](https://api.python.langchain.com/en/latest/chat_models/langchain_groq.chat_models.ChatGroq.html) class as valid dictionary fields."
        st.markdown("Layer Agent Config", help=tooltip)
        # new_layer_agent_config = st_ace(
        #     value=json.dumps(st.session_state.layer_agent_config, indent=2),
        #     language='json',
        #     placeholder="Layer Agent Configuration (JSON):",
        #     show_gutter=False,
        #     wrap=True,
        #     auto_update=True
        # )
        new_layer_agent_config = st.text_area(
            label = "tab1-Layer Agent Configuration (JSON)",
            value=json.dumps(st.session_state.layer_agent_config, indent=2),
            height=400
        )
        
        if st.form_submit_button("Update Configuration"):
            if is_valid_json(new_layer_agent_config):   
                try:
                    new_layer_config = json.loads(new_layer_agent_config)
                    st.session_state.layer_agent_config = new_layer_config
                    set_moa_agent(
                        main_model=new_main_model,
                        cycles=new_cycles,
                        layer_agent_config=new_layer_config,
                        main_model_temperature=main_temperature,
                        override=True
                    )
                    st.session_state.messages = []
                    st.success("Configuration updated successfully!")
                except json.JSONDecodeError:
                    st.error("Invalid JSON in Layer Agent Configuration. Please check your input.")
                except Exception as e:
                    st.error(f"Error updating configuration: {str(e)}")
            else:
                st.error("Invalid JSON in Layer Agent Configuration. Please check your input")

    st.markdown("---")
    st.markdown("""
    ### Credits     
    - LLMs: [Groq](https://groq.com/)  
    - Article: [Medium](https://medium.com/@kram254/running-mixture-of-agents-on-groq-with-streamlit-on-localhost-90b5fec4d35c#:~:text=The%20Mixture%2Dof%2DAgents%20framework,for%20developing%20sophisticated%20AI%20solutions.)              
    """)


with tab2:
    # title of the document comparison section
    st.title(f""":rainbow[Document Comparison]""")

    with st.expander("Current MOA Configuration", expanded=False):
        st.markdown(f"**Main Model**: ``{st.session_state.main_model}``")
        st.markdown(f"**Main Model Temperature**: ``{st.session_state.main_temp:.1f}``")
        st.markdown(f"**Layers**: ``{st.session_state.cycles}``")
        st.markdown(f"**Layer Agents Configuration**:")        
        f_new_layer_agent_config = st.text_area(
            label = "tab4-Layer Agent Configuration (JSON)",
            value=json.dumps(st.session_state.layer_agent_config, indent=2),
            height=400
        )


    # default container that houses the document upload field
    with st.container():
        # header that is shown on the web UI
        st.header(f""":rainbow[Document Upload]""")
        with st.expander("💡 :rainbow[Document Selection from System]"):
            col1, col2 = st.columns(2)
            with col1:
                # the first file upload field, the specific ui element that allows you to upload file 1
                File1 = st.file_uploader('Upload File 1 (Document A)', type=["pdf"], key="doc_1")
            with col2:
                # the second file upload field, the specific ui element that allows you to upload file 2
                File2 = st.file_uploader('Upload File 2 (Document B)', type=["pdf"], key="doc_2")
                # when both files are uploaded it saves the files to the directory, creates a path, and invokes the

        #compare = st.button("Compare Documents", type="primary")

        #if compare:
        # doc_compare Function
        if File1 and File2 is not None:
            # determine the path to temporarily save the PDF file that was uploaded
            #save_folder = os.getenv('save_folder')
            save_folder = r'H:\Interview Preparation\Coding\GenAI\Tryouts\1 - DOCUMENT COMPARISON\46-DocumentComparison\docs'
            # create a posix path of save_folder and the first file name
            save_path_1 = Path(save_folder, File1.name)
            # create a posix path of save_folder and the second file name
            save_path_2 = Path(save_folder, File2.name)
            # write the first uploaded PDF to the save_folder you specified
            # st.write(save_path_1)
            with open(save_path_1, mode='wb') as w:
                w.write(File1.getvalue())
            # write the second uploaded PDF to the save_folder you specified
            with open(save_path_2, mode='wb') as w:
                w.write(File2.getvalue())
            # once the save path exists for both documents you are trying to compare...
            if save_path_1.exists() and save_path_2.exists():
                # write a success message saying the first file has been successfully saved
                st.success(f'File {File1.name} is successfully saved!')
                # write a success message saying the second file has been successfully saved
                st.success(f'File {File2.name} is successfully saved!')
                # running the document comparison task, and outputting the results to the front end
                with st.spinner('Comparing documents...'):
                    changes = doc_compare(save_path_1, save_path_2)
                    st.session_state.doc_changes = changes
                    st.write(changes)
                    # removing the first PDF that was temporarily saved to perform the comparison task
                    os.remove(save_path_1)
                    # removing the second PDF that was temporarily saved to perform the comparison task
                    os.remove(save_path_2)

            with st.spinner("Processing..."):
                if st.session_state.doc_changes:
                    main_decision_model = st.session_state.main_model#"deepseek-r1-distill-llama-70b",
                    query = st.session_state.doc_changes + "Now based on the given ccomparison result generate a final report in a structured manner with tabular format with cotenxtual meaning of each changes:",
                    f_ast_mess = stream_response(chat(
                                query,
                                main_decision_model,
                                SYSTEM_PROMPT, 
                                st.session_state.layer_agent_config,
                                chat_memory,  # The chat_memory object must have load_memory_variables() and save_context() methods
                                cycles = int(st.session_state.cycles),
                                save = True,
                                output_format='json'))            
                
                    st.header("📌 Final Document Comparison Report")
                    # **Step 1: Capture the full output from `st.write_stream()`**
                    full_output = capture_stream(f_ast_mess)

                    # **Step 2: Process the captured output**
                    processed_text, think_contents = process_stream(full_output)               
                    
                    # **Step 3: Display <think> content inside an expander**
                    if st.session_state.main_model == "deepseek-r1-distill-llama-70b":
                        if think_contents:
                            with st.expander("💡 Thinking Process"):
                                for thought in think_contents:
                                    st.markdown(thought, unsafe_allow_html=True)  # Properly formatted Markdown content