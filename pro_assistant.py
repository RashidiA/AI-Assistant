import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg
import google.generativeai as genai
import io
from gtts import gTTS
import base64

# --- 1. SETTINGS & MODEL ---
st.set_page_config(page_title="Abu AI Assistant", page_icon="🤖")
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Setup Gemini 2.5 Flash (2026 Standard)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction="Your name is Abu. You are a friendly Malaysian AI. Speak naturally in Manglish. If the user's input is unclear or just noise, politely ask: 'Sorry, tak dengar lah. Boleh cakap lagi sekali?'"
    )
else:
    st.error("Missing API Key in Secrets!")
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
        # Use unsafe_allow_html for the 2026 Streamlit standard
        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except:
        pass

# --- 2. USER INTERFACE ---
st.title("🤖 Abu: The Manglish Assistant")
st.info("I only respond if you say 'Abu' first. I'll stop recording automatically when you stop talking.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. THE SMART MIC (Noise-Filtering) ---
st.write("---")
audio_bytes = audio_recorder(
    text="Say 'Abu...' and speak",
    recording_color="#e84a5f",
    neutral_color="#6aa36f",
    icon_size="3x",
    energy_threshold=0.01, # Higher = ignores more background noise
    pause_threshold=2.0    # Automatically stops after 2 seconds of silence
)

# --- 4. THE BRAIN ---
if audio_bytes:
    with st.spinner("Abu is listening..."):
        try:
            # Step A: Prepare Audio
            audio_io = io.BytesIO(audio_bytes)
            sound = AudioSegment.from_file(audio_io)
            wav_buffer = io.BytesIO()
            sound.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            # Step B: Transcription (Whisper handles Manglish best)
            r = sr.Recognizer()
            with sr.AudioFile(wav_buffer) as source:
                audio_data = r.record(source)
                user_text = r.recognize_whisper(audio_data, model="tiny")
            
            # Step C: Abu Logic (Wake word check)
            if "abu" in user_text.lower():
                # Extract the actual question after the name
                clean_prompt = user_text.lower().replace("abu", "").strip()
                
                if not clean_prompt:
                    ai_response = "Ya, saya Abu. Ada apa-apa nak tanya?"
                else:
                    response = model.generate_content(clean_prompt)
                    ai_response = response.text
                
                st.session_state.messages.append({"role": "user", "content": user_text})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.rerun()
            
            else:
                # If audio was detected but 'Abu' wasn't said
                st.toast("Did you call Abu? I didn't hear my name.")

        except Exception as e:
            st.error("Wait ah, Abu got confused. Try again?")

# Autoplay AI response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    speak_text(st.session_state.messages[-1]["content"])
