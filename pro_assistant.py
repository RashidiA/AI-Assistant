import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg
import google.generativeai as genai
import io
import os

# --- 1. SETUP & BRAIN CONFIGURATION ---
st.set_page_config(page_title="Gemini Voice Assistant", page_icon="🤖")

# Point Pydub to the portable ffmpeg binary we included in requirements
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Connect to Gemini (Requires GEMINI_API_KEY in Streamlit Secrets)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("Missing API Key! Please add GEMINI_API_KEY to your Streamlit Secrets.")
    st.stop()

# Initialize memory so the assistant remembers the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. USER INTERFACE ---
st.title("🤖 AI Voice Assistant")
st.markdown("---")

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. THE "EARS" (VOICE CAPTURE) ---
with st.sidebar:
    st.header("Voice Control")
    st.write("1. Click 'Start' to turn on mic.")
    st.write("2. Speak your request.")
    st.write("3. Click 'Process' below.")
    
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# --- 4. THE PROCESSING ENGINE ---
if webrtc_ctx.audio_receiver:
    if st.button("🎤 Process Voice Command", use_container_width=True):
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        
        if len(audio_frames) > 0:
            with st.spinner("Transcribing & Thinking..."):
                try:
                    # Stitching recorded audio chunks
                    sound = AudioSegment.empty()
                    for frame in audio_frames:
                        sound += AudioSegment(
                            data=frame.to_ndarray().tobytes(),
                            sample_width=frame.format.bytes,
                            frame_rate=frame.sample_rate,
                            channels=len(frame.layout.channels)
                        )
                    
                    # Convert to WAV for the Speech Recognizer
                    audio_buffer = io.BytesIO()
                    sound.export(audio_buffer, format="wav")
                    audio_buffer.seek(0)
                    
                    # Speech to Text
                    r = sr.Recognizer()
                    with sr.AudioFile(audio_buffer) as source:
                        audio_data = r.record(source)
                        user_text = r.recognize_google(audio_data)
                    
                    # Add User Voice to Chat
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    
                    # Send to Gemini & Get Response
                    chat = model.start_chat(history=[
                        {"role": m["role"], "parts": [m["content"]]} 
                        for m in st.session_state.messages[:-1]
                    ])
                    response = chat.send_message(user_text)
                    ai_response = response.text

                    # Add AI Response to Chat
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                    # Refresh to show new messages
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing audio: {e}")
        else:
            st.warning("No audio detected. Did you speak while the mic was active?")
