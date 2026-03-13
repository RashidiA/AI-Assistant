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

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        system_instruction="You are a helpful assistant in Malaysia. Understand English, Malay, and Manglish. Respond naturally in a friendly tone."
    )
else:
    st.error("Missing API Key in Secrets!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FUNCTION: TEXT TO SPEECH ---
def speak_text(text):
    # Using 'ms' (Malay) as it handles Manglish accents better than 'en'
    tts = gTTS(text=text, lang='ms') 
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    # Create an autoplaying audio player using HTML
    b64 = base64.b64encode(fp.read()).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_markdown=True)

# --- 2. UI ---
st.title("🤖 Talking Voice Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. MIC CONTROL ---
with st.sidebar:
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# --- 4. PROCESSING ---
if webrtc_ctx.audio_receiver:
    if st.button("🚀 Record & Speak"):
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        if len(audio_frames) > 0:
            with st.spinner("Talking to Gemini..."):
                try:
                    # Audio Stitching
                    sound = AudioSegment.empty()
                    for frame in audio_frames:
                        sound += AudioSegment(
                            data=frame.to_ndarray().tobytes(),
                            sample_width=frame.format.bytes,
                            frame_rate=frame.sample_rate,
                            channels=len(frame.layout.channels)
                        )
                    
                    # Transcription
                    audio_buffer = io.BytesIO()
                    sound.export(audio_buffer, format="wav")
                    audio_buffer.seek(0)
                    r = sr.Recognizer()
                    with sr.AudioFile(audio_buffer) as source:
                        audio_data = r.record(source)
                        user_text = r.recognize_whisper(audio_data, model="tiny")
                    
                    # Generate Response
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    response = model.generate_content(user_text)
                    ai_response = response.text
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                    # Output Text and Trigger Speech
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")

# Trigger speech for the very last message if it's from the assistant
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    speak_text(st.session_state.messages[-1]["content"])
