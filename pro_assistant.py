import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg
import google.generativeai as genai
import io
from gtts import gTTS
import base64

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Manglish AI Assistant", page_icon="🤖")

# Fix for ffmpeg
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Setup Gemini
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        system_instruction="You are a friendly Malaysian AI. Understand and respond in Manglish (mixed English and Malay)."
    )
else:
    st.error("Please add GEMINI_API_KEY to Streamlit Secrets!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FUNCTION: SPEAK ---
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='ms') 
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(md, unsafe_markdown=True)
    except:
        pass

# --- 2. USER INTERFACE ---
st.title("🎙️ Talking Manglish AI")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. MIC CONTROL (STUN FIX) ---
with st.sidebar:
    st.header("Mic Control")
    # Using a try-block to ignore the Python 3.14 shutdown bug
    try:
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"video": False, "audio": True},
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]}
                ]
            }
        )
    except AttributeError:
        # This catches the '_polling_thread' error silently
        webrtc_ctx = None
        st.warning("WebRTC initialization issue. Please refresh the page.")

# --- 4. PROCESSING ---
if webrtc_ctx and webrtc_ctx.audio_receiver:
    if st.button("🚀 Process Voice", use_container_width=True):
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        if len(audio_frames) > 0:
            with st.spinner("Decoding Manglish..."):
                try:
                    sound = AudioSegment.empty()
                    for frame in audio_frames:
                        sound += AudioSegment(
                            data=frame.to_ndarray().tobytes(),
                            sample_width=frame.format.bytes,
                            frame_rate=frame.sample_rate,
                            channels=len(frame.layout.channels)
                        )
                    
                    audio_buffer = io.BytesIO()
                    sound.export(audio_buffer, format="wav")
                    audio_buffer.seek(0)
                    
                    r = sr.Recognizer()
                    with sr.AudioFile(audio_buffer) as source:
                        audio_data = r.record(source)
                        user_text = r.recognize_whisper(audio_data, model="tiny")
                    
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    response = model.generate_content(user_text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("No audio detected. Click Start first!")

# Autoplay speech
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    speak_text(st.session_state.messages[-1]["content"])
