import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg
import io
import os

# --- STEP 1: FIX FFMPEG PATH ---
# This points pydub to the standalone ffmpeg binary provided by imageio-ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = ffmpeg_path

st.set_page_config(page_title="AI Assistant", page_icon="🎙️")

## --- UI Layout ---
st.title("🎙️ Voice AI Assistant")
st.markdown("Click **Start** to open the mic, speak, then click **Process Audio**.")

# Initialize Speech Recognizer
r = sr.Recognizer()

# --- STEP 2: AUDIO CAPTURE COMPONENT ---
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# --- STEP 3: PROCESSING LOGIC ---
if webrtc_ctx.audio_receiver:
    if st.button("🚀 Process My Speech"):
        try:
            # Get audio frames from the buffer
            audio_frames = webrtc_ctx.audio_receiver.get_frames()
            
            if len(audio_frames) > 0:
                with st.spinner("Analyzing your voice..."):
                    # Concatenate frames into one AudioSegment
                    sound = AudioSegment.empty()
                    for frame in audio_frames:
                        sound += AudioSegment(
                            data=frame.to_ndarray().tobytes(),
                            sample_width=frame.format.bytes,
                            frame_rate=frame.sample_rate,
                            channels=len(frame.layout.channels)
                        )
                    
                    # Convert to WAV in memory (SpeechRecognition needs WAV/AIFF/FLAC)
                    audio_buffer = io.BytesIO()
                    sound.export(audio_buffer, format="wav")
                    audio_buffer.seek(0)

                    # Recognize using Google Speech API
                    with sr.AudioFile(audio_buffer) as source:
                        audio_data = r.record(source)
                        text = r.recognize_google(audio_data)
                        
                        # Display results
                        st.chat_message("user").write(text)
                        
                        with st.chat_message("assistant"):
                            st.write(f"I understood: **{text}**")
                            st.info("What would you like me to do with this information?")
            else:
                st.warning("No audio frames captured. Did you start the mic and speak?")
                
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Waiting for microphone connection... Click 'Start' above.")
