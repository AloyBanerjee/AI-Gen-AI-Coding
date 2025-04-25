import streamlit as st
import validators
import subprocess
import os
import glob
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from dotenv import load_dotenv
import whisper
from transformers import pipeline
import streamlit.components.v1 as components

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

### Common Function

def download():
    if not video_url:
        st.warning("Please enter a YouTube URL.")
        return
    
    os.makedirs(f"{output_dir}/video", exist_ok=True)
    quality_option = f'-f "bestvideo[height<={video_quality[:-1]}]+bestaudio/best"' if video_quality not in ["best", "worst"] else f'-f "{video_quality}"'
    command = f'yt-dlp {quality_option} -P "{output_dir}/video" "{video_url}"'
    with st.expander('Processing...', expanded=False):
        output = run_command(command)
    st.success("Download completed!")

    # Get the latest downloaded file
    downloaded_files = sorted(glob.glob(f"{output_dir}/video/*"), key=os.path.getctime, reverse=True)
    if downloaded_files:
        st.session_state["latest_file"] = downloaded_files[0]
        st.session_state["downloaded_files"] = downloaded_files
        st.video(downloaded_files[0])

def extract_audio():
    if not video_url:
        st.warning("Please enter a YouTube URL.")
        return
    
    audio_output_dir = f"{output_dir}/audio"
    os.makedirs(audio_output_dir, exist_ok=True)

    # Use -o to specify the filename format
    command = f'yt-dlp -x --audio-format mp3 -o "{audio_output_dir}/%(title)s.%(ext)s" "{video_url}"'
    with st.expander('Processing...', expanded=False):
        output = run_command(command)

    st.success("Audio extracted!")

    # Get the latest downloaded file
    downloaded_files = sorted(glob.glob(f"{output_dir}/audio/*"), key=os.path.getctime, reverse=True)
    if downloaded_files:
        st.session_state["latest_audio_file"] = downloaded_files[0]
        st.session_state["downloaded_audio_files"] = downloaded_files
        st.audio(downloaded_files[0])


def get_metadata():
    if not video_url:
        st.warning("Please enter a YouTube URL.")
        return
    
    command = f'yt-dlp --dump-json "{video_url}"'
    output = run_command(command)

def list_formats():
    if not video_url:
        st.warning("Please enter a YouTube URL.")
        return
    
    command = f'yt-dlp -F "{video_url}"'
    output = run_command(command)

def run_command(command):
    """Executes a shell command and streams the output in a command-line style."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output_text = ""
    progress_placeholder = st.empty()
    loader_placeholder = st.spinner("Processing...")
    with loader_placeholder:
        for line in process.stdout:
            output_text += line
            progress_placeholder.code(output_text, language="bash")
            st.session_state["progress"] = output_text
    
    process.wait()
    return output_text

def summarize(website_url, llm):
    prompt_template="""
        Provide a summary of the following content in {language} within 300 words:
        Content:{text} 
        """
    prompt=PromptTemplate(template=prompt_template,input_variables=["text","language"])
    try:
        with st.spinner(f"Summarization is in progres for {website_url}, please keep waiting..."):
            ## Validate the given url
            if not website_url.strip():
                st.error("Please provide the information to get started")
            elif not validators.url(website_url):
                st.error("Please enter a valid Url. It can may be a YT video utl or website url")
            ## loading the website or yt video data
            if "youtube.com" in website_url:
                loader=YoutubeLoader.from_youtube_url(website_url,add_video_info=True)
                # loader = YoutubeLoader.from_youtube_url(
                #     "https://www.youtube.com/watch?v=QsYGlZkevEg", add_video_info=False
                # )
            else:
                loader=UnstructuredURLLoader(urls=[website_url],ssl_verify=False,
                                            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
            docs=loader.load()                
            if not docs:
                st.error("No content could be loaded from the URL.")
                st.stop()
            
            ## Chain For Summarization
            chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
            output_summary = chain.run(
                input_documents=docs,  # Pass as Document objects
                language="Hindi"       # Pass additional variable
            )
            #output_summary=chain.run({"text": text, "language": "Hindi"})

            st.success(output_summary)
    except Exception as e:
        st.exception(f"Exception:{e}")

def listdowndownloadedvideos(output_dir='downloads/'):
    downloaded_video_files = sorted(glob.glob(f"{output_dir}/video/*"), key=os.path.getctime, reverse=True)
    downloaded_audio_files = sorted(glob.glob(f"{output_dir}/audio/*"), key=os.path.getctime, reverse=True)
    if downloaded_video_files:
        st.session_state["latest_file"] = downloaded_video_files[0]
        st.session_state["downloaded_files"] = downloaded_video_files
    if downloaded_audio_files:
        st.session_state["latest_audio_file"] = downloaded_audio_files[0]
        st.session_state["downloaded_audio_files"] = downloaded_audio_files

def transcribe_audio(file_path):
    model = whisper.load_model("base")  # you can try 'tiny' for faster results
    result = model.transcribe(file_path)
    return result['text']

def summarize_text(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    # Optional: split long text into smaller chunks
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = [summarizer(chunk)[0]['summary_text'] for chunk in chunks]
    
    return "\n".join(summaries)

def summarize_with_ollama(text, trading=True):
    ollama_llm = Ollama(model="llama3.2:latest")  
   
    prompt_template = PromptTemplate(
        input_variables=["transcript"],
        template="""
        Summarize the following transcript into a concise summary with key points:

        Transcript:
        {transcript}

        Summary:
        """
    )
    stock_summary_prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""You are a financial analyst and expert summarizer. Given the following transcript from an audio conversation or presentation, your task is to extract all valuable and actionable insights related to stock trading, investments, and the financial markets.

                Your summary must include the following sections clearly:

                ---

                📌 **1. Stock Mentions and Trading Recommendations**  
                - List all individual stocks, ETFs, or companies mentioned.  
                - Note any price targets, buy/sell/hold recommendations, or trading strategies discussed.  
                - Include any technical indicators or chart patterns referenced (e.g., RSI, MACD, moving averages).  
                - Mention sentiment (bullish, bearish, neutral) if expressed.

                ---

                📈 **2. Market Trends and Economic Commentary**  
                - Summarize any macroeconomic trends discussed (e.g., inflation, interest rates, GDP, Fed policy).  
                - Note commentary about sectors (e.g., tech, healthcare, energy) and their outlooks.  
                - Include geopolitical or regulatory factors affecting markets.

                ---

                💡 **3. Investment Strategies or Tips**  
                - Extract any trading strategies, portfolio advice, or financial planning suggestions.  
                - Note if they mention short-term vs. long-term plays, options trading, or risk management tactics.  
                - Highlight any “golden nuggets” or expert tips mentioned.

                ---

                🧠 **4. Expert Quotes or Standout Insights**  
                - Quote any strong or memorable statements that offer unique insight.  
                - Highlight any conflicting viewpoints or debates discussed.

                ---

                📚 **5. Additional Mentions (optional)**  
                - Cryptocurrency, commodities, real estate, bonds, or other asset classes if referenced.  
                - Tools, apps, brokers, or platforms mentioned.

                ---

                Please structure the output clearly using bullet points or numbered lists where appropriate, and ensure clarity and conciseness.

                Here is the transcript:
                {transcript}
                """
                )
    
    if trading:
        selected_prompt_template = prompt_template
    else:
        selected_prompt_template = prompt_template


    chain = LLMChain(llm=ollama_llm, prompt=selected_prompt_template)
    summary = chain.run(transcript=text)
    return summary


st.set_page_config(
        page_title="StreamSage", page_icon=r"H:\Interview Preparation\Coding\GenAI\Tryouts\45-YoutubeDownloader\downloadingicon.png", layout="wide"
    )

# Streamlit UI
st.title("🎬🎵 StreamSage: The AI-Powered Video Companion for Music & Market Lovers", anchor=False)

st.info("Download YouTube videos, playlists, and extract audio with ease! In case you need summary of the video,"
" you can use the Summarization tab to get the summary of the video content."
)

# Input URL
video_url = st.text_input("Enter YouTube Video/Playlist/Website URL:")

# Select download type
download_type = st.radio("Select Download Type:", ["Single Video", "Playlist"])

# Select output directory
output_dir = st.text_input("Enter Download Location (Default: downloads/):", "downloads")

# Select video quality
video_quality = st.selectbox("Select Video Quality:", ["best", "worst", "1080p", "720p", "480p", "360p"])  

# Buttons for operations
col0, col1, col2, col3, col4, col5, col6, col7 = st.tabs([
    "⚙️ Configuration",
    "📥 Download", 
    "🎵 Extract Audio",
    "📄 Get Metadata",
    "📌 List Available Formats", 
    "▶️ Play Videos",
    "📝 Summarization", 
    "🧠 Stock Insights"
])

with col0:
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
    default_config = st.button('Default Config')
    if default_config:
        if GROQ_API_KEY.strip():
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
            st.success("Groq API Key & large langugae model is successfully set!")
        else:
            st.error("Please provide the api key to get started")
    if groq_api_key and not default_config:
        if groq_api_key.strip():
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
            st.success("Groq API Key & large langugae model is successfully set!")
        else:
            st.error("Please provide the api key to get started")
with col1:
    if st.button("📥 Download Video", type='primary'):
        download()
with col2:
    if st.button("🎵 Extract Audio", type='primary'):
        extract_audio()
with col3:
    if st.button("📄 Get Metadata", type='primary'):
        get_metadata()
with col4:
    if st.button("📌 List Available Formats", type='primary'):
        list_formats()
with col5:
    # Show all downloaded files if available
    listdowndownloadedvideos()
    if "downloaded_files" in st.session_state:
        st.subheader("▶️ Play Downloaded Videos")
        selected_video = st.selectbox("🎬 Select a video to play:", st.session_state["downloaded_files"])
        if selected_video:
            st.video(selected_video, )
with col6:
    listdowndownloadedvideos()
    if "downloaded_files" in st.session_state:
        st.subheader("🦜 Summarize the Youtube Video")
        selected_video = st.selectbox("📝 Select a video to summarize:", st.session_state["downloaded_files"])
        
        if selected_video:
            st.info(selected_video)
            # if st.button("📝 Summerize Content", type='primary'):

            #     audio_path = r"H:\Interview Preparation\Coding\GenAI\Tryouts\45-YoutubeDownloader\downloads\audio\Build AI Assistant With MCP Servers And  Tools Using LangChain And Groq.mp3"

            #     # Loader 1: Transcribe audio
            #     with st.spinner("🔊 Transcribing audio..."):
            #         transcript = transcribe_audio(audio_path)
            #     with st.expander('Transcript', expanded=False):
            #         st.write(transcript)

            #     # Loader 2: Generate intermediate summary
            #     with st.spinner("🧠 Generating intermediate summary..."):
            #         summary = summarize_text(transcript)
            #     with st.expander('Intermediate Summary', expanded=False):
            #         st.write(summary)

            #     # Loader 3: Final summaries (detailed)
            #     with st.spinner("📊 Generating final summaries..."):
            #         final_trading_summary = summarize_with_ollama(transcript, trading=True)
            #         final_generic_summary = summarize_with_ollama(transcript, trading=False)

            #     # Display results
            #     if final_generic_summary and final_trading_summary:
            #         st.subheader("📝 Final Summary")
            #         tab1, tab2 = st.tabs(["📝 Generic Summary", "📈 Stock Trading Advice: Summary"])
            #         with tab1:
            #             st.success(final_generic_summary)
            #         with tab2:
            #             st.info(final_trading_summary)

with col7:    
    st.title("🌐 Get Stock Anlysis")
    components.iframe("https://jyotibansalanalysis.com/", height=600, scrolling=True)