import streamlit as st
import json
from typing import Iterable
from func import *
from streamlit_ace import st_ace
import copy
from io import StringIO
import re
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = OpenAIEmbeddings()
if "retriever" not in st.session_state:
    st.session_state.retriever = None

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
    page_title="Mixture of Experts - Groq",
    page_icon=r'H:\Interview Preparation\Coding\GenAI\Tryouts\41-MixtureofAgent-Notebook\static\favicon.ico',
        menu_items={
        'About': "## Groq Mixture-Of-Agents \n Powered by [Groq](https://groq.com)"
    },
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

set_moa_agent()
   

tab1, tab2, tab3, tab4 = st.tabs([
    "🛠️ MOA Configuration", 
    "📄 Contextual Information", 
    "🤖 Chat Bot", 
    "📋 FDA Reportability Decision"
])

with tab1:
    # config_form = st.form("Agent Configuration", border=False)
    st.title("MOA Configuration")
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
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Contextual Information Loading", anchor=False)
    with col2:
        st.image(r'H:\Interview Preparation\Coding\GenAI\Tryouts\41-MixtureofAgent-Notebook\static\banner1.png', width=500)
        
    with st.form("Contextual Information Loading", border=False):        
        try:
            uploaded_file = st.file_uploader(
                "Upload a file for RAG retrieval", type=["txt", "pdf", "docx"]
            )
            

            if st.form_submit_button("Update Context Information", type="primary"):                
                if uploaded_file is not None:                  
                    try:
                        st.write(uploaded_file.name)
                        file_extension = uploaded_file.name.split(".")[-1]                    
                        st.write(file_extension)
                        # Save uploaded file temporarily
                        temp_file_path = os.path.join("temp", uploaded_file.name)
                        os.makedirs("temp", exist_ok=True)

                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        if file_extension == "pdf":
                            loader = PyPDFLoader(temp_file_path)
                            st.write('Loaded PDF')
                        elif file_extension == "docx":
                            loader = Docx2txtLoader(temp_file_path)
                        else:
                            loader = TextLoader(temp_file_path)
                        
                        documents = loader.load()

                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=100
                        )
                        splitted_document = text_splitter.split_documents(documents)
                        st.write(f"Number of chunks: {len(splitted_document)}")

                        # Initialize FAISS index if not present
                        index_path = "faiss_index"
                        embedding_model = OpenAIEmbeddings()
                        st.session_state.embedding_model = embedding_model

                        db = FAISS.from_documents(splitted_document,embedding=embedding_model)
                        if os.path.exists(index_path):
                            db.load_local(index_path, embeddings=embedding_model, allow_dangerous_deserialization=True) 
                        else:
                            os.makedirs(index_path, exist_ok=True)
                            db.save_local(index_path)
                        
                        st.session_state.db = db        
                        retriever = db.as_retriever()
                        st.session_state.retriever = retriever        

                        st.success("RAG Configuration updated successfully!")

                    except Exception as e:
                        st.error(f"Error configuring RAG details: {str(e)}")
        except Exception as e:
            st.error(f"Error configuring RAG details: {str(e)}")
  
  
with tab3:
    # Main app layout
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Mojo - Mixture of Expert Agents", anchor=False)
    with col2:
       st.image(r'H:\Interview Preparation\Coding\GenAI\Tryouts\41-MixtureofAgent-Notebook\static\banner1.png', width=500)

    st.write("Mixture of Agents architecture Powered by Groq LLMs.")
       
    # Display current configuration
    with st.expander("Current MOA Configuration", expanded=False):
        st.markdown(f"**Main Model**: ``{st.session_state.main_model}``")
        st.markdown(f"**Main Model Temperature**: ``{st.session_state.main_temp:.1f}``")
        st.markdown(f"**Layers**: ``{st.session_state.cycles}``")
        st.markdown(f"**Layer Agents Config**:")
        # new_layer_agent_config = st_ace(
        #     value=json.dumps(st.session_state.layer_agent_config, indent=2),
        #     language='json',
        #     placeholder="Layer Agent Configuration (JSON)",
        #     show_gutter=False,
        #     wrap=True,
        #     readonly=True,
        #     auto_update=True
        # )
        new_layer_agent_config = st.text_area(
            label = "tab3-Layer Agent Configuration (JSON)",
            value=json.dumps(st.session_state.layer_agent_config, indent=2),
            height=400
        )

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    use_rag = st.toggle("Use RAG Agent", value=False)

    if query := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        if use_rag:
            # Retrieve from RAG
            retriever = st.session_state.retriever
            db = st.session_state.db 
            ast_mess = stream_response(RAG_chat(
                                            retriever,
                                            query,
                                            st.session_state.main_model,
                                            SYSTEM_PROMPT,
                                            st.session_state.layer_agent_config,
                                            chat_memory,  # The chat_memory object must have load_memory_variables() and save_context() methods
                                            cycles = int(st.session_state.cycles),
                                            save = True,
                                            output_format='json'))
            response = st.write_stream(ast_mess)
        else:
            moa_agent = st.session_state.moa_agent
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                #ast_mess = stream_response(moa_agent.chat(query, output_format='json'))
                ast_mess = stream_response(chat(
                                            query,
                                            st.session_state.main_model,
                                            SYSTEM_PROMPT,
                                            st.session_state.layer_agent_config,
                                            chat_memory,  # The chat_memory object must have load_memory_variables() and save_context() methods
                                            cycles = int(st.session_state.cycles),
                                            save = True,
                                            output_format='json'))

                response = st.write_stream(ast_mess)
    
        st.session_state.messages.append({"role": "assistant", "content": response})


with tab4:
    fcol1, fcol2 = st.columns([2, 1])
    with fcol1:
        st.title("FDA Reportability Event Identification", anchor=False)
    with fcol2:
       st.image(r'H:\Interview Preparation\Coding\GenAI\Tryouts\41-MixtureofAgent-Notebook\static\banner1.png', width=500)

    st.write("Mixture of Agents architecture Powered by Groq LLMs.")   
   
    # Display current configuration
    with st.expander("FDA - Current MOA Configuration", expanded=False):
        st.markdown(f"**Main Model**: ``{st.session_state.main_model}``")
        st.markdown(f"**Main Model Temperature**: ``{st.session_state.main_temp:.1f}``")
        st.markdown(f"**Layers**: ``{st.session_state.cycles}``")
        st.markdown(f"**Layer Agents Configuration**:")        
        f_new_layer_agent_config = st.text_area(
            label = "tab4-Layer Agent Configuration (JSON)",
            value=json.dumps(st.session_state.layer_agent_config, indent=2),
            height=400
        )

    q_col, ans_col = st.columns([1, 2])
    with q_col:
        st.header("Query : What is the FDA Reportability Decision for this event?")
        query = st.text_area(label="Enter the event details here", height=200)
        get_event_type = st.button("Process the Event", type="primary")        
        if get_event_type:
            with st.spinner("Processing..."):
                f_ast_mess = stream_response(chat(
                                                query + "Now based on the given context finally decide what would be the event type among Death, Injury & Malfunction:",
                                                "deepseek-r1-distill-llama-70b",
                                                SYSTEM_PROMPT, 
                                                st.session_state.layer_agent_config,
                                                chat_memory,  # The chat_memory object must have load_memory_variables() and save_context() methods
                                                cycles = int(st.session_state.cycles),
                                                save = True,
                                                output_format='json'))
                
                
                #event_response = st.write_stream(f_ast_mess)
                st.header("📌 Event Type")
                # **Step 1: Capture the full output from `st.write_stream()`**
                full_output = capture_stream(f_ast_mess)

                # **Step 2: Process the captured output**
                processed_text, think_contents = process_stream(full_output)               

                # **Step 4: Display <think> content inside an expander**
                if think_contents:
                    with st.expander("💡 Thinking Process"):
                        for thought in think_contents:
                            st.markdown(thought, unsafe_allow_html=True)  # Properly formatted Markdown

                

    with ans_col:
        if get_event_type:
            moa_agent = st.session_state.moa_agent
            #st.write(json.dumps(st.session_state.layer_agent_config, indent=2)) 
            spinner_placeholder = st.empty()
            with st.spinner("Processing..."):
                f_ast_mess = stream_response(chat(
                                            query + "Now based on the given context finally decide whether the given event is reportable or not and also explain in details:",
                                            st.session_state.main_model,
                                            SYSTEM_PROMPT, 
                                            st.session_state.layer_agent_config,
                                            chat_memory,  # The chat_memory object must have load_memory_variables() and save_context() methods
                                            cycles = int(st.session_state.cycles),
                                            save = True,
                                            output_format='json'))
            
            spinner_placeholder.empty()            
            f_response = st.write_stream(f_ast_mess)
            st.header("FDA Reportability Decision")
            st.write(f_response)