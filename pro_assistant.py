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
        system_instruction="""Your name is Abu. You are a local Malaysian expert. 
        You understand that voice transcriptions might be slightly wrong (e.g., 'Toulon Shari' means 'Tolong cari'). 
        Always respond in natural Manglish. If you are really stuck, ask: 'Sorry boss, tak clear lah. Boleh repeat?'"""
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
st.info("Cakap 'Abu' then your request. Example: 'Abu, tolong cari tempat makan kat Tanjung Malim.'")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. THE MIC (Optimized for Noisy Environments) ---
audio_bytes = audio_recorder(
    text="Click & speak (Say 'Abu...')",
    recording_color="#e84a5f",
    neutral_color="#6aa36f",
    icon_size="3x",
    energy_threshold=0.02, # Increased to ignore more background hum
    pause_threshold=2.5    # Give you more time to breathe between words
)

# --- 4. THE BRAIN ---
if audio_bytes:
    with st.spinner("Abu is processing..."):
        try:
            audio_io = io.BytesIO(audio_bytes)
            sound = AudioSegment.from_file(audio_io)
            wav_buffer = io.BytesIO()
            sound.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            r = sr.Recognizer()
            with sr.AudioFile(wav_buffer) as source:
                audio_data = r.record(source)
                
                # IMPROVED: Adding a 'prompt' tells Whisper what words to expect
                user_text = r.recognize_whisper(
                    audio_data, 
                    model="tiny", 
                    language="ms",
                    initial_prompt="Abu, tolong cari, tempat makan, sedap, kat, Tanjung Malim, Manglish"
                )
            
            st.write(f"🔍 Abu heard: *{user_text}*")

            # Fuzzy name check to prevent "Toulon" vs "Abu" confusion
            wake_words = ["abu", "abo", "aboo", "ah bu", "toulon", "arbu"]
            found_wake_word = any(word in user_text.lower() for word in wake_words)

            if found_wake_word:
                # Clean the prompt
                clean_prompt = user_text.lower()
                for word in wake_words:
                    clean_prompt = clean_prompt.replace(word, "")
                
                # If it misheard 'Tolong cari' as 'Toulon Shari', we pass the whole thing
                final_input = user_text if len(clean_prompt) < 2 else clean_prompt
                
                response = model.generate_content(final_input)
                ai_response = response.text
                
                st.session_state.messages.append({"role": "user", "content": user_text})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.rerun()
            else:
                st.toast("I heard you, but say 'Abu' first lah!")

        except Exception as e:
            st.error("Ayo, something went wrong. Try again?")

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    speak_text(st.session_state.messages[-1]["content"])
