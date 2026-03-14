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
st.set_page_config(page_title="Abu Assistant", page_icon="🤖")
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        # STRICTOR INSTRUCTIONS FOR LESS TALKING
        system_instruction="Your name is Abu. Be extremely brief. Answer in Manglish using 15 words or less. No small talk."
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
st.title("🤖 Abu (Short & Sweet)")

# Only show the last 3 messages to keep it clean
for message in st.session_state.messages[-3:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. THE MIC (TIGHTENED SENSITIVITY) ---
audio_bytes = audio_recorder(
    text="Click to speak to Abu",
    recording_color="#e84a5f",
    neutral_color="#6aa36f",
    icon_size="2x",
    energy_threshold=0.05, # HIGHER = Ignores more background noise
    pause_threshold=1.5    # Stops faster when you finish speaking
)

# --- 4. PROCESSING ---
if audio_bytes:
    try:
        audio_io = io.BytesIO(audio_bytes)
        sound = AudioSegment.from_file(audio_io)
        wav_buffer = io.BytesIO()
        sound.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        r = sr.Recognizer()
        with sr.AudioFile(wav_buffer) as source:
            audio_data = r.record(source)
            user_text = r.recognize_whisper(audio_data, model="base", language="ms")
        
        # Check for Abu/Abo/Arbu
        if any(w in user_text.lower() for w in ["abu", "abo", "arbu"]):
            prompt = user_text.lower().replace("abu", "").replace("abo", "").strip()
            
            if len(prompt) > 2:
                try:
                    response = model.generate_content(prompt)
                    ai_response = response.text
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.rerun()
                except Exception as api_err:
                    if "429" in str(api_err):
                        st.toast("Wait 30s ah, Abu tired.")
            else:
                st.toast("Ya, Abu here. What you want?")
    except Exception as e:
        pass # Silently ignore errors to stop "too much talking" from the app

# Autoplay
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    speak_text(st.session_state.messages[-1]["content"])
