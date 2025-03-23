import streamlit as st
import subprocess
import os
import glob


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


st.set_page_config(
        page_title="Video Downloader - Companion for Music Lover", page_icon="H:\Interview Preparation\Coding\GenAI\Tryouts\45-YoutubeDownloader\music.webp", layout="wide"
    )

# Streamlit UI
col1, col2 = st.columns([0.5, 10])
with col1:
    st.image(r'H:\Interview Preparation\Coding\GenAI\Tryouts\45-YoutubeDownloader\downloadicon.png')    
with col2:
    st.title("Video Downloader - Companion for Music Lover", anchor=False)



# Input URL
video_url = st.text_input("Enter YouTube Video/Playlist URL:")

# Select download type
download_type = st.radio("Select Download Type:", ["Single Video", "Playlist"])

# Select output directory
output_dir = st.text_input("Enter Download Location (Default: downloads/):", "downloads")

# Select video quality
video_quality = st.selectbox("Select Video Quality:", ["best", "worst", "1080p", "720p", "480p", "360p"])  

# Buttons for operations
col1, col2, col3, col4, col5 = st.tabs([
    "📥 Download", 
    "🎵 Extract Audio",
    "📄 Get Metadata",
    "📌 List Available Formats", 
    "▶️ Play Videos"
])


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
