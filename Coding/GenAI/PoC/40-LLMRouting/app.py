import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['TOGETHERAI_API_KEY'] = os.getenv("TOGETHER_API_KEY") 

from routellm.controller import Controller


# ["mf"]: Model Fusion (Combining models intelligently).
# ["rr"]: Round-Robin (Alternates between models).
# ["latency"]: Chooses the fastest available model.
# ["ensemble"]: Runs multiple models and combines results.
# ["custom_router"]: A user-defined routing strategy.

# Initialize RouteLLM client
client = Controller(
    routers=["mf"],
    strong_model="gpt-4o-mini",
    weak_model= "together_ai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    # [
    #     "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    #     "together_ai/mistral/Mistral-7B-Instruct-v0.2",       
    #     "together_ai/qwen/Qwen-QwQ-32B-Preview",
    #     "together_ai/gemma/Gemma-2-Instruct-27B",
    #     "together_ai/flux/FLUX-1-schnell"
    # ]
)



st.set_page_config(page_title="LLM Routing",page_icon=r"H:\Interview Preparation\Coding\GenAI\Tryouts\40-LLMRouting\routellmicon.png", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .chat-container {
        background-color: #2b2d42;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .user-bubble {
        background-color: #8d99ae;
        color: white;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 70%;
        align-self: flex-end;
    }
    .ai-bubble {
        background-color: aliceblue;
        color: black;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 100%;
    }
    .model-info {
        color: black;
        font-size: 14px;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Set up the application


col1, col2 = st.columns([3, 1])
with col1:
    st.header("AI Conversation Hub with Routing", divider="red")   
with col2:              
    st.image(r"H:\Interview Preparation\Coding\GenAI\Tryouts\40-LLMRouting\routellmicon.png", use_container_width=False)
    
st.markdown("Powered by RouterLLM | Intelligent Model Routing")

# Initialize conversation history
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Display conversation
chat_display = st.container()
with chat_display:
    for entry in st.session_state.chat_log:
        if entry["sender"] == "user":
            st.markdown(
                f'<div class="user-bubble">{entry["text"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="ai-bubble">{entry["text"]}</div>',
                unsafe_allow_html=True
            )
            if "ai_model" in entry:
                st.caption(f'Powered by:{entry["ai_model"]}')
                # st.markdown(
                #     f'<div class="model-info"></div>',
                #     unsafe_allow_html=True
                # )

# Input section
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Type your message here...",
        height=100,
        key="input_field"
    )
    submit_button = st.form_submit_button(label="Send", type="primary")

# Process input and generate response
if submit_button and user_input:
    # Add user's message to conversation
    st.session_state.chat_log.append({"sender": "user", "text": user_input})
    
    # Generate AI response
    with st.spinner("AI is thinking..."):
        ai_response = client.chat.completions.create(
            model="router-mf-0.11593",
            messages=[{"role": "user", "content": user_input}]
        )
        response_text = ai_response['choices'][0]['message']['content']
        model_version = ai_response['model']
        
        # Add AI response to conversation
        st.session_state.chat_log.append({
            "sender": "assistant",
            "text": response_text,
            "ai_model": model_version
        })
    
    # Refresh the page to show new messages
    st.rerun()

# Add a clear chat button
if st.button("Clear Conversation", type="secondary"):
    st.session_state.chat_log = []
    st.rerun()