import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import pandas as pd
import numpy as np
from scipy import stats
from pptx import Presentation
from pptx.util import Inches
import os
import glob
import queue
import pydub
from io import BytesIO
from duckduckgo_search import DDGS
import speech_recognition as sr
from datetime import datetime

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Rashidi Engineering Assistant", layout="wide")

# --- 1. VOICE PROCESSING (WEB-READY) ---
def process_audio_to_text(audio_frames):
    """Converts browser audio frames into text using Google Speech API."""
    if not audio_frames:
        return None
    
    sound = pydub.AudioSegment.empty()
    for frame in audio_frames:
        s = pydub.AudioSegment(
            data=frame.to_ndarray().tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels)
        )
        sound += s
    
    # Export to memory buffer
    buffer = BytesIO()
    sound.export(buffer, format="wav")
    buffer.seek(0)
    
    r = sr.Recognizer()
    with sr.AudioFile(buffer) as source:
        audio_data = r.record(source)
        try:
            return r.recognize_google(audio_data)
        except:
            return "Command not recognized."

# --- 2. ENGINEERING DATA TOOLS ---
def analyze_data(file_path):
    df = pd.read_csv(file_path)
    data = df.select_dtypes(include=[np.number]).iloc[:, 0] # First numeric column
    mean = data.mean()
    std = data.std()
    cpk = min((10.5 - mean)/(3*std), (mean - 9.5)/(3*std)) # Default LSL/USL
    return mean, cpk

def web_research(query):
    with DDGS() as ddgs:
        return [r for r in ddgs.text(f"engineering standard {query}", max_results=3)]

# --- 3. UI LAYOUT ---
st.title("🤖 Pro AI Engineering Assistant")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("🎙️ Voice Command")
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
    )
    
    status_placeholder = st.empty()
    if webrtc_ctx.state.playing:
        status_placeholder.info("Listening... click STOP when finished.")
        # Logic to pull frames from the receiver
        frames = []
        try:
            while True:
                frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                frames.append(frame)
        except:
            pass
        
        if not webrtc_ctx.state.playing and frames:
            user_text = process_audio_to_text(frames)
            st.session_state['last_cmd'] = user_text

with col2:
    st.header("⌨️ Command Terminal")
    user_input = st.text_input("Type or use Voice:", value=st.session_state.get('last_cmd', ""))
    
    if user_input:
        cmd = user_input.lower()
        
        # ACTION: ANALYSIS
        if "analyze" in cmd or "ppt" in cmd:
            csv_files = glob.glob("*.csv")
            if csv_files:
                target = csv_files[0]
                mean, cpk = analyze_data(target)
                st.success(f"Analyzed {target}: Mean={mean:.2f}, Cpk={cpk:.2f}")
                
                # Create PPT
                prs = Presentation()
                slide = prs.slides.add_slide(prs.slide_layouts[1])
                slide.shapes.title.text = f"Quality Report: {target}"
                slide.placeholders[1].text = f"Analyzed on {datetime.now()}\nMean: {mean}\nCpk: {cpk}"
                
                ppt_io = BytesIO()
                prs.save(ppt_io)
                st.download_button("📥 Download PowerPoint", ppt_io.getvalue(), "Report.pptx")
            else:
                st.warning("No CSV files found for analysis.")

        # ACTION: WEB SEARCH
        elif "search" in cmd:
            query = cmd.replace("search", "")
            results = web_research(query)
            for r in results:
                st.write(f"🔗 [{r['title']}]({r['href']})")
                st.caption(r['body'])

# --- SIDEBAR ---
st.sidebar.image("")
st.sidebar.title("Help")
st.sidebar.info("""
**Try saying:**
* "Analyze welding data"
* "Search ISO 9001 standards"
* "Make a PPT report"
""")