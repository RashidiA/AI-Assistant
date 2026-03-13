import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import io

# 1. Setup the Speech Recognizer
r = sr.Recognizer()

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = io.BytesIO()

    def recv(self, frame):
        # This converts the browser's WebRTC audio frame to an AudioSegment
        sound = AudioSegment(
            data=frame.to_ndarray().tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels)
        )
        # Exporting to a buffer to act like a file
        sound.export(self.audio_buffer, format="wav")
        return frame

def main():
    st.title("🎙️ AI Voice Assistant")
    st.write("No PyAudio/ALSA needed! This uses WebRTC + Pydub.")

    # 2. The WebRTC UI Component
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
    )

    if webrtc_ctx.audio_receiver:
        if st.button("Process Last Audio"):
            try:
                # Get the audio frames from the receiver
                audio_frames = webrtc_ctx.audio_receiver.get_frames()
                if len(audio_frames) > 0:
                    # Combine frames into one segment
                    sound = AudioSegment.empty()
                    for frame in audio_frames:
                        wframe = AudioSegment(
                            data=frame.to_ndarray().tobytes(),
                            sample_width=frame.format.bytes,
                            frame_rate=frame.sample_rate,
                            channels=len(frame.layout.channels)
                        )
                        sound += wframe
                    
                    # Convert to WAV for SpeechRecognition
                    audio_data_io = io.BytesIO()
                    sound.export(audio_data_io, format="wav")
                    audio_data_io.seek(0)

                    with sr.AudioFile(audio_data_io) as source:
                        audio = r.record(source)
                        text = r.recognize_google(audio)
                        st.success(f"Recognized: {text}")
                else:
                    st.warning("No audio frames captured yet. Talk into the mic!")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
