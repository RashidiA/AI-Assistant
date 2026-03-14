import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg
import google.generativeai as genai
import io
from gtts import gTTS
import base64

# --- 1. SETUP ---
st.set_page_config(page_title="Abu AI Assistant", page_icon="🤖")
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Setup Gemini 2.5 Flash (Standard for 2026)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction="Your name is Abu. You are a local expert in Malaysia. Answer questions directly and helpfully in Manglish."
    )
else:
    st.error("Missing API Key!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

def speak_text(text):
    try:
        tts = gTTS(text=text, lang='ms')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)
    except:
        pass

# --- 2. UI ---
st.title("🤖 Abu Assistant")
st.info("Say 'Abu' clearly, then ask your question. Example: 'Abu, suggest lunch spots nearby.'")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. THE MIC (Optimized Settings) ---
audio_bytes = audio_recorder(
    text="Click, say 'Abu...', then ask",
    recording_color="#e84a5f",
    neutral_color="#6aa36f",
    icon_size="3x",
    energy_threshold=0.01, # Lower threshold to catch softer voices
    pause_threshold=3.0    # Give you 3 seconds to finish your sentence
)

# --- 4. PROCESSING ---
if audio_bytes:
    with st.spinner("Abu is thinking..."):
        try:
            audio_io = io.BytesIO(audio_bytes)
            sound = AudioSegment.from_file(audio_io)
            wav_buffer = io.BytesIO()
            sound.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            r = sr.Recognizer()
            with sr.AudioFile(wav_buffer) as source:
                audio_data = r.record(source)
                # FIX: We use 'base' model for better accuracy if 'tiny' is confusing
                user_text = r.recognize_whisper(audio_data, model="base", language="ms")
            
            st.write(f"🔍 Abu heard: *{user_text}*")

            # Check if Abu was called
            if "abu" in user_text.lower() or "abo" in user_text.lower():
                # Get the actual question
                prompt = user_text.lower().replace("abu", "").replace("abo", "").strip()
                
                # If there's a question, send to Gemini
                if len(prompt) > 2:
                    response = model.generate_content(prompt)
                    ai_response = response.text
                    
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.rerun()
                else:
                    st.warning("I heard my name, but what is your question, boss?")
            else:
                st.toast("Please start with 'Abu' so I know you are talking to me!")

        except Exception as e:
            st.error(f"Ayo, something went wrong: {e}")

# Autoplay
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    speak_text(st.session_state.messages[-1]["content"])
