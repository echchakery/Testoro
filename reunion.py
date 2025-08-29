# reunion.py (or /pages/📁 Audio Note & Transcription.py)
# 📁 reunion.py
import streamlit as st
import json
import tempfile
import shutil
import os
import subprocess
from datetime import datetime
import whisper
import requests
import pandas as pd
import streamlit as st
# 📁 pages/📁 Audio Note & Transcription.py
import streamlit as st
import json
import tempfile
import shutil
import os
import subprocess
from datetime import datetime
import whisper
import requests
import pandas as pd

# ---------------- CONFIG ----------------
LLM_API_KEY = "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"
LLM_MODEL_ID = "llama-3.1-8b-instant"
LLM_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# ---------------- Whisper ----------------
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")
    result = model.transcribe(audio_path, language="fr")
    return result["text"]

# ---------------- Video to Audio Conversion ----------------
def convert_to_wav(input_file, output_file="converted_audio.wav"):
    cmd = [
        "ffmpeg", "-i", input_file,
        "-vn",  # remove video
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_file
    ]
    subprocess.run(cmd, check=True)
    return output_file

# ---------------- Groq LLaMA Summarizer ----------------
def summarize_with_llama(transcript_text):
    prompt = f"Résume cette transcription de réunion en points clés clairs et concis :\n\n{transcript_text}"

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL_ID,
        "messages": [
            {"role": "system", "content": "Tu es un assistant expert en résumé de réunions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }

    resp = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ---------------- Main render function ----------------
def render():
    st.set_page_config(page_title="📁 Audio Transcription", layout="wide")

    st.title("📁 Audio & Meeting Transcription")
    st.markdown("Transforme tes notes vocales ou vidéos de réunion en **transcriptions claires et résumés concis.**")

    # Tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🎧 Upload/Record", "📝 Transcription", "📄 Summary", "📜 History"])

    audio_file, wav_audio_data = None, None
    transcription, summary = None, None

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📂 Upload Audio/Video")
            audio_file = st.file_uploader(
                "Upload audio/video (MP3, WAV, M4A, MP4, WEBM, MKV)",
                type=["mp3", "wav", "m4a", "mp4", "webm", "mkv"]
            )
        with col2:
            st.subheader("🎤 Record Audio")
            try:
                from st_audiorec import st_audiorec
                wav_audio_data = st_audiorec()
            except:
                st.warning("⚠️ Install st-audiorec to enable recording.")

    # Processing
    if audio_file or wav_audio_data:
        try:
            temp_dir = tempfile.gettempdir()

            if audio_file:
                suffix = os.path.splitext(audio_file.name)[1]
                temp_path = os.path.join(temp_dir, f"uploaded_audio{suffix}")
                with open(temp_path, "wb") as f:
                    shutil.copyfileobj(audio_file, f)

                if suffix.lower() in [".mp4", ".webm", ".mkv"]:
                    temp_path = convert_to_wav(temp_path, os.path.join(temp_dir, "converted_audio.wav"))

            elif wav_audio_data:
                temp_path = os.path.join(temp_dir, "recorded_audio.wav")
                with open(temp_path, "wb") as f:
                    f.write(wav_audio_data)

            # Transcription
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(temp_path)

            # Résumé
            with st.spinner("Summarizing with LLaMA..."):
                summary = summarize_with_llama(transcription)

            # Save to history
            os.makedirs("history", exist_ok=True)
            transcription_file = "history/audio_transcriptions.json"
            if os.path.exists(transcription_file):
                with open(transcription_file, "r", encoding="utf-8") as f:
                    log = json.load(f)
            else:
                log = []

            log.append({
                "timestamp": datetime.now().isoformat(),
                "filename": audio_file.name if audio_file else "recorded_audio.wav",
                "transcription": transcription,
                "summary": summary
            })

            with open(transcription_file, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2, ensure_ascii=False)

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # Display transcription
    with tab2:
        if transcription:
            st.success("✅ Transcription complete!")
            st.text_area("📝 Transcribed Text", transcription, height=250)

    # Display summary
    with tab3:
        if summary:
            st.subheader("📄 Meeting Summary")
            st.info(summary)
            st.download_button(
                label="⬇️ Download Summary (TXT)",
                data=summary,
                file_name="meeting_summary.txt",
                mime="text/plain"
            )

    # History
    with tab4:
        if os.path.exists("history/audio_transcriptions.json"):
            with open("history/audio_transcriptions.json", "r", encoding="utf-8") as f:
                logs = json.load(f)
            if logs:
                df = pd.DataFrame(logs)
                st.dataframe(df[["timestamp", "filename", "summary"]].tail(5))
            else:
                st.info("No transcription history yet.")
        else:
            st.info("No transcription history found.")

"""
def render():
    st.title("📁 Réunion")
    st.write("Contenu réunion ici.")
    # ...

# ---------------- CONFIG ----------------
LLM_API_KEY = "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"
LLM_MODEL_ID = "llama-3.1-8b-instant"
LLM_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# ---------------- Whisper ----------------
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")
    result = model.transcribe(audio_path, language="fr")
    return result["text"]

# ---------------- Video to Audio Conversion ----------------
def convert_to_wav(input_file, output_file="converted_audio.wav"):
    cmd = [
        "ffmpeg", "-i", input_file,
        "-vn",  # remove video
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_file
    ]
    subprocess.run(cmd, check=True)
    return output_file

# ---------------- Groq LLaMA Summarizer ----------------
def summarize_with_llama(transcript_text):
    prompt = f"Résume cette transcription de réunion en points clés clairs et concis :\n\n{transcript_text}"

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL_ID,
        "messages": [
            {"role": "system", "content": "Tu es un assistant expert en résumé de réunions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }

    resp = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📁 Audio Transcription", layout="wide")

st.title("📁 Audio & Meeting Transcription")
st.markdown("Transforme tes notes vocales ou vidéos de réunion en **transcriptions claires et résumés concis.**")

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["🎧 Upload/Record", "📝 Transcription", "📄 Summary", "📜 History"])

audio_file, wav_audio_data = None, None
transcription, summary = None, None

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📂 Upload Audio/Video")
        audio_file = st.file_uploader(
            "Upload audio/video (MP3, WAV, M4A, MP4, WEBM, MKV)",
            type=["mp3", "wav", "m4a", "mp4", "webm", "mkv"]
        )
    with col2:
        st.subheader("🎤 Record Audio")
        try:
            from st_audiorec import st_audiorec
            wav_audio_data = st_audiorec()
        except:
            st.warning("⚠️ Install st-audiorec to enable recording.")

# Processing
if audio_file or wav_audio_data:
    try:
        temp_dir = tempfile.gettempdir()

        if audio_file:
            suffix = os.path.splitext(audio_file.name)[1]
            temp_path = os.path.join(temp_dir, f"uploaded_audio{suffix}")
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(audio_file, f)

            if suffix.lower() in [".mp4", ".webm", ".mkv"]:
                temp_path = convert_to_wav(temp_path, os.path.join(temp_dir, "converted_audio.wav"))

        elif wav_audio_data:
            temp_path = os.path.join(temp_dir, "recorded_audio.wav")
            with open(temp_path, "wb") as f:
                f.write(wav_audio_data)

        # Transcription
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(temp_path)

        # Résumé
        with st.spinner("Summarizing with LLaMA..."):
            summary = summarize_with_llama(transcription)

        # Save to history
        os.makedirs("history", exist_ok=True)
        transcription_file = "history/audio_transcriptions.json"
        if os.path.exists(transcription_file):
            with open(transcription_file, "r", encoding="utf-8") as f:
                log = json.load(f)
        else:
            log = []

        log.append({
            "timestamp": datetime.now().isoformat(),
            "filename": audio_file.name if audio_file else "recorded_audio.wav",
            "transcription": transcription,
            "summary": summary
        })

        with open(transcription_file, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Display transcription
with tab2:
    if transcription:
        st.success("✅ Transcription complete!")
        st.text_area("📝 Transcribed Text", transcription, height=250)

# Display summary
with tab3:
    if summary:
        st.subheader("📄 Meeting Summary")
        st.info(summary)
        st.download_button(
            label="⬇️ Download Summary (TXT)",
            data=summary,
            file_name="meeting_summary.txt",
            mime="text/plain"
        )

# History
with tab4:
    if os.path.exists("history/audio_transcriptions.json"):
        with open("history/audio_transcriptions.json", "r", encoding="utf-8") as f:
            logs = json.load(f)
        if logs:
            df = pd.DataFrame(logs)
            st.dataframe(df[["timestamp", "filename", "summary"]].tail(5))
        else:
            st.info("No transcription history yet.")
    else:
        st.info("No transcription history found.")
"""
