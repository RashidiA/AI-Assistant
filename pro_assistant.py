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

if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction="Your name is Abu. You are a friendly Malaysian assistant. Speak naturally in Manglish. If the user's input is unclear, politely ask for clarification."
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
st.title("🤖 Abu: The Manglish Assistant")
st.info("Say 'Abu' clearly. Tip: Abu hears better if there is no background noise!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. THE MIC (Adjusted Sensitivity) ---
audio_bytes = audio_recorder(
    text="Say 'Abu...' and speak",
    recording_color="#e84a5f",
    neutral_color="#6aa36f",
    icon_size="3x",
    energy_threshold=0.015, # Slightly higher to ignore small background noises
    pause_threshold=2.0 
)

# --- 4. THE BRAIN ---
if audio_bytes:
    with st.spinner("Abu is listening..."):
        try:
            audio_io = io.BytesIO(audio_bytes)
            sound = AudioSegment.from_file(audio_io)
            wav_buffer = io.BytesIO()
            sound.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            r = sr.Recognizer()
            with sr.AudioFile(wav_buffer) as source:
                audio_data = r.record(source)
                # FIX: Set language to 'ms' (Malay) to improve name recognition
                user_text = r.recognize_whisper(audio_data, model="tiny", language="ms")
            
            # Show what Abu heard for debugging
            st.write(f"🔍 Abu heard: *{user_text}*")

            # FIX: Fuzzy matching for the name 'Abu'
            wake_words = ["abu", "abo", "aboo", "ah bu"]
            found_wake_word = any(word in user_text.lower() for word in wake_words)

            if found_wake_word:
                # Remove the wake word from the prompt
                clean_prompt = user_text.lower()
                for word in wake_words:
                    clean_prompt = clean_prompt.replace(word, "")
                clean_prompt = clean_prompt.strip()
                
                if not clean_prompt:
                    ai_response = "Ya, saya Abu. Nak tanya apa?"
                else:
                    response = model.generate_content(clean_prompt)
                    ai_response = response.text
                
                st.session_state.messages.append({"role": "user", "content": user_text})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.rerun()
            else:
                st.toast("I didn't hear my name (Abu). Try again?")

        except Exception as e:
            st.error("Wait ah, Abu got confused. Try again?")

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    speak_text(st.session_state.messages[-1]["content"])
