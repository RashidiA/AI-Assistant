import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg
import google.generativeai as genai
import io
from gtts import gTTS
import base64

# --- 1. INITIAL SETUP ---
st.set_page_config(page_title="2026 Manglish AI", page_icon="🇲🇾", layout="centered")
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Connect to Gemini 2.5 (Current 2026 Standard)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # System instruction for natural Manglish behavior
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction="You are a helpful assistant in Malaysia. Understand and respond in Manglish (mixed English/Malay) naturally. Use 'lah', 'leh', and 'can' where appropriate."
    )
else:
    st.error("Missing GEMINI_API_KEY! Add it to your Streamlit Secrets.")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- VOICE OUTPUT FUNCTION ---
def speak_text(text):
    try:
        # 'ms' language handles the Malaysian accent for Manglish perfectly
        tts = gTTS(text=text, lang='ms') 
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        # The FIX: unsafe_allow_html=True is required for autoplay
        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Voice playback failed: {e}")

# --- 2. USER INTERFACE ---
st.title("🎙️ Talking Manglish AI")
st.caption("2026 Edition • Powered by Gemini 2.5 Flash")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. AUDIO INPUT (The Stable Way) ---
st.write("---")
st.write("Click the mic, speak, then click again to finish:")
audio_bytes = audio_recorder(
    text="",
    recording_color="#ff4b4b",
    neutral_color="#999999",
    icon_size="3x",
)

# --- 4. THE BRAIN ---
if audio_bytes:
    with st.spinner("Wait ah, I listening..."):
        try:
            # Step A: Prepare Audio
            audio_io = io.BytesIO(audio_bytes)
            sound = AudioSegment.from_file(audio_io)
            
            wav_buffer = io.BytesIO()
            sound.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            # Step B: Transcription (Whisper is great for mixed languages)
            r = sr.Recognizer()
            with sr.AudioFile(wav_buffer) as source:
                audio_data = r.record(source)
                user_text = r.recognize_whisper(audio_data, model="tiny")
            
            # Step C: Generate AI Response
            st.session_state.messages.append({"role": "user", "content": user_text})
            
            response = model.generate_content(user_text)
            ai_response = response.text
            
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            # Refresh to show text and trigger speech
            st.rerun()

        except Exception as e:
            st.error(f"Ayo, error pula: {e}")

# Trigger speech if the last message is from the AI
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    speak_text(st.session_state.messages[-1]["content"])
