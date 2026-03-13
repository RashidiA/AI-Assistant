import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg
import google.generativeai as genai
import io
import os

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Manglish AI Assistant", page_icon="🇲🇾")

# Point Pydub to the portable ffmpeg
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Setup Gemini
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        system_instruction="You are a helpful assistant in Malaysia. Understand English, Malay, and Manglish. Respond naturally."
    )
else:
    st.error("Add your GEMINI_API_KEY to Streamlit Secrets!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. USER INTERFACE ---
st.title("🎙️ Manglish Voice AI")
st.caption("Recognizes English, Malay, and Mixed languages.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. THE MIC (WebRTC with STUN Fix) ---
with st.sidebar:
    st.header("Mic Control")
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        # THE FIX FOR "CONNECTING":
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        }
    )

# --- 4. PROCESSING ---
if webrtc_ctx.audio_receiver:
    if st.button("🚀 Process Voice", use_container_width=True):
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        
        if len(audio_frames) > 0:
            with st.spinner("Processing Manglish..."):
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
                        # Uses local Whisper Tiny (multilingual)
                        user_text = r.recognize_whisper(audio_data, model="tiny")
                    
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    response = model.generate_content(user_text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")
