import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg
import google.generativeai as genai
import io
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Manglish AI Assistant", page_icon="🇲🇾")

# Fix for ffmpeg on Streamlit Cloud
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Setup Gemini with a "Manglish" personality
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # System instruction ensures the AI understands and responds to mixed languages
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        system_instruction="You are a helpful assistant in Malaysia. You understand English, Malay, and Manglish (mixed). Respond in the same style the user uses."
    )
else:
    st.error("Missing API Key! Please add GEMINI_API_KEY to your Streamlit Secrets.")
    st.stop()

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. USER INTERFACE ---
st.title("🎙️ Manglish Voice Assistant")
st.write("Cakap je apa-apa in English or Malay (or both!)")

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. AUDIO CAPTURE (SIDEBAR) ---
with st.sidebar:
    st.header("Controls")
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# --- 4. THE BRAIN (WHISPER + GEMINI) ---
if webrtc_ctx.audio_receiver:
    if st.button("🚀 Record & Process", use_container_width=True):
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        
        if len(audio_frames) > 0:
            with st.spinner("Decoding mixed language..."):
                try:
                    # Combine audio chunks
                    sound = AudioSegment.empty()
                    for frame in audio_frames:
                        sound += AudioSegment(
                            data=frame.to_ndarray().tobytes(),
                            sample_width=frame.format.bytes,
                            frame_rate=frame.sample_rate,
                            channels=len(frame.layout.channels)
                        )
                    
                    # Convert to WAV in memory
                    audio_buffer = io.BytesIO()
                    sound.export(audio_buffer, format="wav")
                    audio_buffer.seek(0)
                    
                    # Multilingual Transcription using Whisper Tiny
                    r = sr.Recognizer()
                    with sr.AudioFile(audio_buffer) as source:
                        audio_data = r.record(source)
                        # recognize_whisper is much better for mixed languages than recognize_google
                        user_text = r.recognize_whisper(audio_data, model="tiny")
                    
                    # Add User Voice to Chat
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    
                    # Send to Gemini
                    response = model.generate_content(user_text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    
                    # Refresh to show new messages
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("No audio detected. Please click 'Start' and speak.")
