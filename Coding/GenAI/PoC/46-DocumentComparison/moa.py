from typing import Generator, Dict, Optional, Literal, TypedDict, List, Union, Iterable
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSerializable
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain.memory import ConversationBufferMemory
import copy
import json


## VALID MODELS & MODEL DROPDOWN CONFIGURATION
valid_main_model= {
    'gemma2-9b-it',
    'llama-guard-3-8b',
    'deepseek-r1-distill-llama-70b'  
}

valid_main_model_list= [
    'gemma2-9b-it',
    'llama-guard-3-8b',
    'deepseek-r1-distill-llama-70b'  
]

valid_model_names = Literal[
    'gemma2-9b-it',
    'llama-guard-3-8b',
    'deepseek-r1-distill-llama-70b'
]

## COMMON VARIABLES
main_model = "gemma2-9b-it"
cycles = 3
main_temp = 0.1
chat_memory = ConversationBufferMemory(
    memory_key="messages",
    return_messages=True
)

## PROMPT
SYSTEM_PROMPT = """\
You are a personal assistant that is helpful.
{helper_response}\
"""
REFERENCE_SYSTEM_PROMPT = """\
You have been provided with a set of responses from various open-source models to the latest user query. 
Your task is to synthesize these responses into a single, high-quality response. 
It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. 
Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. 
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Responses from models:
{responses}
"""
reference_system_prompt = REFERENCE_SYSTEM_PROMPT

## DEFAULT CONFIG
default_config = {
    "main_model": "gemma2-9b-it",
    "cycles": 3,
    "layer_agent_config": {},
}
layer_agent_config_def = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "gemma2-9b-it"
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "gemma2-9b-it",
        "temperature": 0.7
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "gemma2-9b-it"
    },
}

## RECOMMENDED CONFIG
rec_config = {
    "main_model": "gemma2-9b-it",
    "cycles": 2,
    "layer_agent_config": {}
}
layer_agent_config_rec = {
    "layer_agent_1": {
        "system_prompt": "Think through your response step by step. {helper_response}",
        "model_name": "gemma2-9b-it",
        "temperature": 0.1,
    },
    "layer_agent_2": {
        "system_prompt": "Respond with a thought and then your response to the question. {helper_response}",
        "model_name": "gemma2-9b-it",
        "temperature": 0.2,
    },
    "layer_agent_3": {
        "system_prompt": "You are an expert at logic and reasoning. Always take a logical approach to the answer. {helper_response}",
        "model_name": "gemma2-9b-it",
        "temperature": 0.4,
    },
    "layer_agent_4": {
        "system_prompt": "You are an expert planner agent. Create a plan for how to answer the human's query. {helper_response}",
        "model_name": "deepseek-r1-distill-llama-70b",
        "temperature": 0.5,
    },
}

## LAYER AGENT CONFIG
layer_agent_config = None
if not layer_agent_config:
    layer_agent_config = {
        'layer_agent_1' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'gemma2-9b-it'},
        'layer_agent_2' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'gemma2-9b-it'},
        'layer_agent_3' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'llama-guard-3-8b'},
        'layer_agent_4' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'deepseek-r1-distill-llama-70b'}
    }

## MAIN DATATYPE CONFIG
class ResponseChunk(TypedDict):
    delta: str
    response_type: Literal['intermediate', 'output']
    metadata: Dict = {}

# COMMON FUNCTIONALITES
def concat_response(
    inputs: Dict[str, str],
    reference_system_prompt: Optional[str] = None
):
    reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT

    responses = ""
    res_list = []
    for i, out in enumerate(inputs.values()):
        responses += f"{i}. {out}\n"
        res_list.append(out)

    formatted_prompt = reference_system_prompt.format(responses=responses)
    return {
        'formatted_response': formatted_prompt,
        'responses': res_list
    }

def create_chain_from_system_prompt(
        system_prompt: str = SYSTEM_PROMPT,
        model_name: str = "gemma2-9b-it",
        **llm_kwargs
    ) -> RunnableSerializable[Dict, str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages", optional=True),
            ("human", "{input}")
        ])

        assert 'helper_response' in prompt.input_variables
        llm = ChatGroq(model=model_name, **llm_kwargs)
        
        chain = prompt | llm | StrOutputParser()
        return chain

def configure_layer_agent(
        layer_agent_config: Optional[Dict] = None
    ) -> RunnableSerializable[Dict, Dict]:
        if not layer_agent_config:
            layer_agent_config = {
                'layer_agent_1' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'gemma2-9b-it'},
                'layer_agent_2' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'gemma2-9b-it'},
                'layer_agent_3' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'gemma2-9b-it'},
                'layer_agent_4' : {'system_prompt': SYSTEM_PROMPT, 'model_name': 'deepseek-r1-distill-llama-70b'}
            }

        parallel_chain_map = dict()
        for key, value in layer_agent_config.items():
            chain = create_chain_from_system_prompt(
                system_prompt=value.pop("system_prompt", SYSTEM_PROMPT), 
                model_name=value.pop("model_name", 'gemma2-9b-it'),
                **value
            )
            parallel_chain_map[key] = RunnablePassthrough() | chain
        
        chain = parallel_chain_map | RunnableLambda(concat_response)
        return chain

def chat(
    input: str,
    main_model: str,
    system_prompt: str,
    layer_agent_config: dict,
    chat_memory,  # The chat_memory object must have load_memory_variables() and save_context() methods    
    cycles: int = 1,
    save: bool = True,
    output_format: Literal['string', 'json'] = 'string',
    messages: Optional[List[BaseMessage]] = None
) -> Generator[str | ResponseChunk, None, None]:
    """
    Process chat requests with layered agents.

    Parameters:
        input (str): The user input.
        main_model (str): The name of the main model to use.
        system_prompt (str): The system prompt to initialize the main agent.
        layer_agent_config (dict): Configuration for the layer agent.
        chat_memory: Object handling message memory, must implement load_memory_variables() and save_context().
        messages (Optional[List[BaseMessage]]): Optional pre-loaded messages.
        cycles (int): Number of cycles to run the layer agent.
        save (bool): Whether to save the conversation to memory.
        output_format (Literal['string', 'json']): Output format type.

    Yields:
        Either a string (if output_format is 'string') or ResponseChunk objects (if output_format is 'json')
    """
    main_agent_chain = create_chain_from_system_prompt(
        system_prompt=system_prompt,
        model_name=main_model
    )

    layer_agent = configure_layer_agent(layer_agent_config)
    
    # Use provided messages or load from memory
    initial_messages = messages or chat_memory.load_memory_variables({}).get('messages', [])
    llm_inp = {
        'input': input,
        'messages': initial_messages,
        'helper_response': ""
    }
    
    # Run layer agent cycles
    for cyc in range(cycles):
        layer_output = layer_agent.invoke(llm_inp)
        l_frm_resp = layer_output['formatted_response']
        l_resps = layer_output['responses']

        # Update input for the main agent using the new helper response
        updated_messages = chat_memory.load_memory_variables({}).get('messages', [])
        llm_inp = {
            'input': input,
            'messages': updated_messages,
            'helper_response': l_frm_resp
        }

        if output_format == 'json':
            for l_out in l_resps:
                yield ResponseChunk(
                    delta=l_out,
                    response_type='intermediate',
                    metadata={'layer': cyc + 1}
                )

    # Stream final output from the main agent chain
    stream = main_agent_chain.stream(llm_inp)
    response = ""
    for chunk in stream:
        if output_format == 'json':
            yield ResponseChunk(
                delta=chunk,
                response_type='output',
                metadata={}
            )
        else:
            yield chunk
        response += chunk

    if save:
        chat_memory.save_context({'input': input}, {'output': response})

def from_config(
    main_model: Optional[valid_model_names] = 'gemma2-9b-it',
    system_prompt: Optional[str] = None,
    cycles: int = 1,
    layer_agent_config: Optional[Dict] = None,
    reference_system_prompt: Optional[str] = None,
    **main_model_kwargs
):
    reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT
    system_prompt = system_prompt or SYSTEM_PROMPT
    layer_agent = configure_layer_agent(layer_agent_config)
    main_agent = create_chain_from_system_prompt(
        system_prompt=system_prompt,
        model_name=main_model,
        **main_model_kwargs
    )
    return {
        "main_agent": main_agent,
        "layer_agent": layer_agent,
        "reference_system_prompt": reference_system_prompt,
        "cycles": cycles
    }

### RAG CHAT
def retrieve_context(retriever,query: str) -> str:
    """Retrieves relevant context from the retriever."""        
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

def RAG_chat(retriever, 
             input,
             main_model: str,
             system_prompt: str,
             layer_agent_config: dict,
             chat_memory,  # The chat_memory object must have load_memory_variables() and save_context() methods
             messages: Optional[List[BaseMessage]] = None,
             cycles: int = 1,
             save: bool = True,
             output_format: Literal['string', 'json'] = 'string') -> Generator[str | ResponseChunk, None, None]:
    """Handles RAG-based text generation."""
    context = retrieve_context(retriever,input)
    enriched_input = f"{input}\n\nContext:\n{context}"
    yield from chat(enriched_input,  
                    main_model,
                    system_prompt,
                    layer_agent_config,
                    chat_memory,
                    None,
                    cycles,
                    save,
                    output_format)

### REASONING RESPONSE HANDLING
def format_reasoning_response(thinking_content):
    """Format assistant content by removing think tags."""
    return (
        thinking_content.replace("<think>\n\n</think>", "")
        .replace("<think>", "")
        .replace("</think>", "")
    )

def is_valid_json(input_string):
    try:
        json.loads(input_string)
        return True
    except json.JSONDecodeError:
        return False