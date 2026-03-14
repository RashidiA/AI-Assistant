import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr # This is what caused your error
from pydub import AudioSegment
import imageio_ffmpeg
import google.generativeai as genai
import io
from gtts import gTTS
import base64

# --- 1. SETUP ---
st.set_page_config(page_title="2026 Manglish AI", page_icon="🇲🇾")
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Connect to Gemini 2.5 (Standard for 2026)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction="You are a friendly Malaysian AI. Use Manglish (mixed English/Malay) naturally."
    )
else:
    st.error("Add GEMINI_API_KEY to your Streamlit Secrets!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- VOICE OUTPUT ---
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='ms') 
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        # Corrected: unsafe_allow_html=True
        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except:
        pass

# --- 2. UI ---
st.title("🎙️ Talking Manglish AI")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. STABLE MIC ---
st.write("Click the mic to speak:")
audio_bytes = audio_recorder(text="", recording_color="#ff4b4b", icon_size="3x")

# --- 4. PROCESSING ---
if audio_bytes:
    with st.spinner("Decoding your Manglish..."):
        try:
            audio_io = io.BytesIO(audio_bytes)
            sound = AudioSegment.from_file(audio_io)
            
            wav_buffer = io.BytesIO()
            sound.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            r = sr.Recognizer()
            with sr.AudioFile(wav_buffer) as source:
                audio_data = r.record(source)
                # Whisper is best for mixed languages
                user_text = r.recognize_whisper(audio_data, model="tiny")
            
            st.session_state.messages.append({"role": "user", "content": user_text})
            response = model.generate_content(user_text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    speak_text(st.session_state.messages[-1]["content"])
