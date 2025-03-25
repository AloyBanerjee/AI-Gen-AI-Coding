import streamlit as st
import validators
import subprocess
import os
import glob
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

### Common Function

def download():
    if not video_url:
        st.warning("Please enter a YouTube URL.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    quality_option = f'-f "bestvideo[height<={video_quality[:-1]}]+bestaudio/best"' if video_quality not in ["best", "worst"] else f'-f "{video_quality}"'
    command = f'yt-dlp {quality_option} -P "{output_dir}" "{video_url}"'
    output = run_command(command)
    st.success("Download completed!")

    # Get the latest downloaded file
    downloaded_files = sorted(glob.glob(f"{output_dir}/*"), key=os.path.getctime, reverse=True)
    if downloaded_files:
        st.session_state["latest_file"] = downloaded_files[0]
        st.session_state["downloaded_files"] = downloaded_files
        st.video(downloaded_files[0])

def extract_audio_old():
    if not video_url:
        st.warning("Please enter a YouTube URL.")
        return
    
    os.makedirs(f"{output_dir}/audio", exist_ok=True)
    command = f'yt-dlp -x --audio-format mp3 -P f"{output_dir}/audio" "{video_url}"'
    output = run_command(command)
    st.success("Audio extracted!")

def extract_audio():
    if not video_url:
        st.warning("Please enter a YouTube URL.")
        return
    
    audio_output_dir = f"{output_dir}/audio"
    os.makedirs(audio_output_dir, exist_ok=True)

    # Use -o to specify the filename format
    command = f'yt-dlp -x --audio-format mp3 -o "{audio_output_dir}/%(title)s.%(ext)s" "{video_url}"'
    output = run_command(command)

    st.success("Audio extracted!")

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
                #loader=YoutubeLoader.from_youtube_url(website_url,add_video_info=True)
                loader = YoutubeLoader.from_youtube_url(
                    "https://www.youtube.com/watch?v=QsYGlZkevEg", add_video_info=False
                )
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


st.set_page_config(
        page_title="Video Downloader - Companion for Music Lover", page_icon="H:\Interview Preparation\Coding\GenAI\Tryouts\45-YoutubeDownloader\music.webp", layout="wide"
    )

# Streamlit UI
col1, col2 = st.columns([0.5, 10])
with col1:
    st.image(r'H:\Interview Preparation\Coding\GenAI\Tryouts\45-YoutubeDownloader\downloadicon.png')    
with col2:
    st.title("Video Downloader - Companion for Music Lover", anchor=False)

st.info("Download YouTube videos, playlists, and extract audio with ease! In case you need summary of the video,"
" you can use the Summarization tab to get the summary of the video content."
" You can also place any website url to get the summary of the content.")

# Input URL
video_url = st.text_input("Enter YouTube Video/Playlist/Website URL:")

# Select download type
download_type = st.radio("Select Download Type:", ["Single Video", "Playlist"])

# Select output directory
output_dir = st.text_input("Enter Download Location (Default: downloads/):", "downloads")

# Select video quality
video_quality = st.selectbox("Select Video Quality:", ["best", "worst", "1080p", "720p", "480p", "360p"])  

# Buttons for operations
col0, col1, col2, col3, col4, col5, col6 = st.tabs([
    "⚙️ Configuration",
    "📥 Download", 
    "🎵 Extract Audio",
    "📄 Get Metadata",
    "📌 List Available Formats", 
    "▶️ Play Videos",
    "📝 Summarization"
])

with col0:
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
    if groq_api_key:
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
    if "downloaded_files" in st.session_state:
        st.subheader("▶️ Play Downloaded Videos")
        selected_video = st.selectbox("🎬 Select a video to play:", st.session_state["downloaded_files"])
        if selected_video:
            st.video(selected_video, )
with col6:
    st.title("🦜 Summarize Text From YT or Website")
    if st.button("📝 Summerize Content", type='primary'):
        #summarize(video_url, llm)
        st.warning('This feature is under development. Please check back later.')