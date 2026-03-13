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
st.set_page_config(page_title="2026 Manglish AI", page_icon="🇲🇾")
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Setup Gemini with 2026 Model
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    try:
        # UPDATED MODEL FOR 2026
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash', 
            system_instruction="You are a friendly Malaysian AI. Use Manglish (mixed English/Malay) naturally."
        )
    except Exception as e:
        st.error(f"Model initialization failed: {e}")
        st.stop()
else:
    st.error("Missing API Key!")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. UI ---
st.title("🎙️ Manglish Assistant v2026")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. STABLE MIC RECORDER ---
st.write("Click the mic to speak (Manglish supported!):")
audio_bytes = audio_recorder(
    text="",
    recording_color="#e84a5f",
    neutral_color="#2a363b",
    icon_size="3x",
)

# --- 4. PROCESSING ---
if audio_bytes:
    with st.spinner("Analyzing audio..."):
        try:
            audio_io = io.BytesIO(audio_bytes)
            sound = AudioSegment.from_file(audio_io)
            
            wav_buffer = io.BytesIO()
            sound.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            r = sr.Recognizer()
            with sr.AudioFile(wav_buffer) as source:
                audio_data = r.record(source)
                # Whisper is still the gold standard for mixed language in 2026
                user_text = r.recognize_whisper(audio_data, model="tiny")
            
            # Save User Input
            st.session_state.messages.append({"role": "user", "content": user_text})
            
            # Get AI Response
            response = model.generate_content(user_text)
            ai_text = response.text
            st.session_state.messages.append({"role": "assistant", "content": ai_text})
            
            # Text to Speech
            tts = gTTS(text=ai_text, lang='ms')
            tts_buffer = io.BytesIO()
            tts.write_to_fp(tts_buffer)
            tts_buffer.seek(0)
            b64 = base64.b64encode(tts_buffer.read()).decode()
            st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_markdown=True)
            
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
