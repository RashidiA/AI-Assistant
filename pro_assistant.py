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
st.set_page_config(page_title="Talking Manglish AI", page_icon="🤖")
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Setup Gemini
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        system_instruction="You are a helpful assistant in Malaysia. Understand English, Malay, and Manglish. Respond naturally in a friendly tone."
    )
else:
    st.error("Missing API Key! Please add GEMINI_API_KEY to your Streamlit Secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FUNCTION: TEXT TO SPEECH ---
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='ms') 
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(md, unsafe_markdown=True)
    except Exception as e:
        st.warning(f"Audio output failed: {e}")

# --- 2. UI ---
st.title("🤖 Talking Voice Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. MIC CONTROL (With Python 3.14 Fixes) ---
with st.sidebar:
    st.header("Mic Control")
    try:
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"video": False, "audio": True},
            # Enhanced RTC configuration to bypass firewalls
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                ]
            }
        )
    except Exception as e:
        st.error(f"WebRTC Error: {e}. Try refreshing the page.")
        webrtc_ctx = None

# --- 4. PROCESSING ---
if webrtc_ctx and webrtc_ctx.audio_receiver:
    if st.button("🚀 Record & Speak", use_container_width=True):
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        if len(audio_frames) > 0:
            with st.spinner("Decoding your voice..."):
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
                        # Whisper is best for Manglish
                        user_text = r.recognize_whisper(audio_data, model="tiny")
                    
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    response = model.generate_content(user_text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing Error: {e}")
        else:
            st.warning("No audio detected. Make sure to click 'Start' and allow microphone access.")

# Trigger speech for the last AI message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    speak_text(st.session_state.messages[-1]["content"])
