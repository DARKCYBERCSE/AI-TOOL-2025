# app.py - ClariView: AI Image Interpreter with Full Feature Set
# Run with: streamlit run app.py

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import io
import re
import asyncio
from gtts import gTTS
from googletrans import Translator
import google.generativeai as genai
import base64
import os

# =============================
# 🔑 Configure GenAI API Key
# =============================
genai.configure(api_key="AIzaSyDhKdVDiV681N9xXy_m5KWh9rM8Kj41B2Q")

# =============================
# 🧠 Initialize Model
# =============================
def initialize_model():
    generation_config = {"temperature": 0.8}
    return genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# =============================
# 🖼️ Generate Content from Image + Questions
# =============================
def generate_content(model, image_path, questions):
    image_part = {
        "mime_type": "image/jpeg",
        "data": image_path.read_bytes()
    }
    results = []
    url_pattern = re.compile(r'(https?://[^\s]+)')

    for question_text in questions:
        question_parts = [question_text, image_part]
        response = model.generate_content(question_parts)

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                parts_output = []
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        text = part.text
                        urls = url_pattern.findall(text)

                        # Replace links in markdown style
                        for url in urls:
                            if any(url.lower().endswith(ext) for ext in [".jpg", ".png", ".jpeg", ".gif"]):
                                parts_output.append(f"🖼️ Image: {url}")
                            elif "youtube.com" in url or "youtu.be" in url:
                                parts_output.append(f"📹 Video: {url}")
                            elif url.lower().endswith((".mp4", ".mov", ".avi")):
                                parts_output.append(f"🎬 Video File: {url}")
                            else:
                                parts_output.append(f"🔗 Link: [{url}]({url})")

                        # Append clean text
                        parts_output.append(text)

                final_output = "<br>".join(parts_output)
                results.append(f"**Q:** {question_text}<br>**A:** {final_output}")
            else:
                results.append(f"**Q:** {question_text}<br>**A:** No content parts found.")
        else:
            results.append(f"**Q:** {question_text}<br>**A:** No candidates found.")
    return results

# =============================
# 🧹 Clean Text (Remove HTML/Markdown)
# =============================
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("**", "")
    text = text.replace("<br>", "\n")
    return text.strip()

# =============================
# 🌍 Translate Text
# =============================
def translate_text(text, lang):
    translator = Translator()
    cleaned = clean_text(text)

    async def run_translation():
        return await translator.translate(cleaned, dest=lang)

    try:
        translation = asyncio.run(run_translation())
        return translation.text
    except Exception as e:
        return f"⚠️ Translation failed: {e}"

# =============================
# 🔊 Generate Audio (TTS)
# =============================
def generate_audio(text, lang):
    tts = gTTS(text, lang=lang)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# =============================
# 🎨 Custom CSS Themes
# =============================
def add_custom_css(mode="dark"):
    if mode == "neon":  # Neon Purple
        st.markdown("""
        <style>
        html, body, .stApp { height: 100%; margin: 0; padding: 0; background: linear-gradient(135deg, #dcdcdc, #f0f0f0, #e6e6fa); color: #2e2e2e; font-family: 'Poppins', sans-serif; }
        .animated-title { font-size: 42px; font-weight: bold; text-align: center; background: linear-gradient(90deg, #8e2de2, #c471ed, #f64f59); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: neonpulse 3s infinite alternate; text-shadow: 0 0 20px #8e2de2, 0 0 40px #c471ed; }
        @keyframes neonpulse { from { text-shadow: 0 0 15px #8e2de2; } to { text-shadow: 0 0 45px #c471ed, 0 0 70px #f64f59; } }
        .chat-bubble { background: #f7f7f7; color: #2e2e2e; padding: 15px; margin: 10px 0; border-radius: 14px; border: 2px solid #8e2de2; box-shadow: 0 0 15px rgba(142,45,226,0.6); animation: fadeInUp 0.8s ease; }
        button[data-testid="baseButton-primary"] { background: linear-gradient(90deg, #8e2de2, #c471ed); color: white !important; font-weight: bold; border-radius: 14px; border: none; transition: 0.3s; box-shadow: 0 0 20px #c471ed; }
        button[data-testid="baseButton-primary"]:hover { background: linear-gradient(90deg, #f64f59, #c471ed); transform: scale(1.08); box-shadow: 0 0 35px #f64f59; }
        textarea { background: #fff !important; color: #2e2e2e !important; border: 2px solid #8e2de2 !important; border-radius: 12px !important; font-family: 'Courier New', monospace; }
        section[data-testid="stSidebar"] { background: #ececec; color: #2e2e2e; border-right: 2px solid #c471ed; }
        </style>""", unsafe_allow_html=True)

    elif mode == "light":
        st.markdown("""
        <style>
        html, body, .stApp { background: #F9F9F9; color: #222; font-family: 'Poppins', sans-serif; }
        .animated-title { font-size: 40px; font-weight: bold; text-align: center; background: linear-gradient(90deg, #FF0077, #FF00FF, #FF0077); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: glowtext 3s infinite alternate; text-shadow: 0 0 20px #FF0077, 0 0 40px #FF00FF; }
        @keyframes glowtext { from { text-shadow: 0 0 10px #FF0077; } to { text-shadow: 0 0 30px #FF00FF, 0 0 60px #FF00FF; } }
        .chat-bubble { background: #FFF; color: #222; padding: 15px; margin: 10px 0; border-radius: 12px; border: 1px solid #FF00FF; box-shadow: 0 0 10px rgba(255,0,150,0.6); animation: fadeInUp 0.6s ease; }
        button[data-testid="baseButton-primary"] { background: #FFF !important; color: #222 !important; font-weight: bold; border-radius: 12px; border: 2px solid #FF00FF; transition: 0.3s; box-shadow: 0 0 10px #FF00FF; }
        button[data-testid="baseButton-primary"]:hover { background: #FF00FF !important; color: white !important; transform: scale(1.05); box-shadow: 0 0 25px #FF00FF; }
        </style>""", unsafe_allow_html=True)

    elif mode == "green-neon":
        st.markdown("""
        <style>
        html, body, .stApp { background: #0d0d0d; color: #00ff99; font-family: 'Courier New', monospace; }
        .animated-title { font-size: 42px; text-align: center; font-weight: bold; background: linear-gradient(90deg, #00ff99, #00ffcc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 25px #00ff99, 0 0 45px #00ffcc; }
        .chat-bubble { background: #1a1a1a; color: #00ffcc; padding: 15px; border-radius: 12px; border: 1px solid #00ff99; box-shadow: 0 0 20px rgba(0,255,153,0.7); }
        </style>""", unsafe_allow_html=True)

    elif mode == "blue-neon":
        st.markdown("""
        <style>
        html, body, .stApp { background: #0d1b2a; color: #00cfff; font-family: 'Poppins', sans-serif; }
        .animated-title { font-size: 42px; font-weight: bold; text-align: center; background: linear-gradient(90deg, #00cfff, #0077ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 25px #00cfff, 0 0 45px #0077ff; }
        .chat-bubble { background: #1a2634; color: #00cfff; padding: 15px; border-radius: 12px; border: 1px solid #0077ff; box-shadow: 0 0 20px rgba(0,153,255,0.7); }
        </style>""", unsafe_allow_html=True)

    elif mode == "dark":
        st.markdown("""
        <style>
        html, body, .stApp { background: #121212; color: #e0e0e0; font-family: 'Roboto', sans-serif; }
        .animated-title { font-size: 40px; font-weight: bold; text-align: center; color: #ffffff; text-shadow: 0 0 15px #00ffcc; }
        .chat-bubble { background: #1e1e1e; color: #e0e0e0; padding: 15px; border-radius: 12px; border: 1px solid #333; box-shadow: 0 0 15px rgba(255,255,255,0.2); }
        </style>""", unsafe_allow_html=True)

    elif mode == "vscode":
        st.markdown("""
        <style>
        html, body, .stApp { background: #1e1e1e; color: #d4d4d4; font-family: 'Consolas', monospace; }
        .animated-title { font-size: 40px; font-weight: bold; text-align: center; color: #569cd6; text-shadow: 0 0 10px #007acc; }
        .chat-bubble { background: #252526; color: #dcdcdc; padding: 15px; border-radius: 12px; border: 1px solid #007acc; box-shadow: 0 0 15px rgba(86,156,214,0.6); }
        </style>""", unsafe_allow_html=True)

# =============================
# 🔍 AI FEATURES
# =============================

def suggest_questions(model, image_path):
    prompt = "Suggest 5 diverse questions about this image. One per line, no numbering."
    image_part = {"mime_type": "image/jpeg", "data": image_path.read_bytes()}
    response = model.generate_content([prompt, image_part])
    if response.candidates and response.candidates[0].content.parts:
        text = response.candidates[0].content.parts[0].text.strip()
        return [q.strip() for q in text.split('\n') if q.strip()]
    return ["What objects are here?", "Describe the background.", "What's the mood?", "Any people present?", "Any text visible?"]

def translate_to_english(text):
    translator = Translator()
    try:
        result = translator.translate(text, dest='en')
        return result.text, result.src
    except Exception as e:
        st.warning(f"⚠️ Translation failed: {e}")
        return text, 'en'

def generate_summary(model, image_path):
    prompt = "Provide a short, natural language summary of this image in one paragraph."
    image_part = {"mime_type": "image/jpeg", "data": image_path.read_bytes()}
    response = model.generate_content([prompt, image_part])
    if response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text.strip()
    return "No summary generated."

def generate_hashtags(model, description):
    prompt = f"Extract 8 social-friendly hashtags from: {description}"
    response = model.generate_content([prompt])
    if response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text.strip()
    return "#AI #ImageAnalysis"

# =============================
# 🎨 UI/UX FEATURES
# =============================

def drag_and_drop_upload():
    return st.file_uploader("📁 Or drag & drop image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

def voice_input():
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Speak now...")
            audio = r.listen(source, timeout=5)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            st.warning("🔇 Could not understand.")
        except sr.RequestError:
            st.error("🌐 Speech service error.")
    except ImportError:
        st.warning("⚠️ Install: `pip install speech_recognition pyaudio`")
    return None

# =============================
# 🎧 AUDIO/TRANSLATION ENHANCEMENTS
# =============================

def batch_translate_and_speak(text, languages):
    translations = {}
    for lang in languages:
        translated = translate_text(text, lang)
        audio = generate_audio(translated, lang)
        translations[lang] = {"text": translated, "audio": audio}
    return translations

# =============================
# 📂 HISTORY & EXPORT
# =============================

def search_history(query):
    results = []
    for idx, entry in enumerate(st.session_state.history):
        combined = " ".join([clean_text(r) for r in entry["results"]]).lower()
        if query.lower() in combined:
            results.append((idx, entry))
    return results

def export_to_pdf(text, filename="ClariView_Log.pdf"):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(0, 10, txt=line, ln=True)
    pdf.output(filename)
    return filename

def export_to_word(text, filename="ClariView_Log.docx"):
    from docx import Document
    doc = Document()
    for line in text.split('\n'):
        doc.add_paragraph(line)
    doc.save(filename)
    return filename

def generate_qr_code(data, filename="qrcode.png"):
    import qrcode
    img = qrcode.make(data)
    img.save(filename)
    return filename

# =============================
# 🔒 OTHER FEATURES
# =============================

def highlight_objects(image_path):
    import cv2
    import numpy as np
    from PIL import Image
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), len(faces)

def generate_meme_caption(model, image_path):
    prompt = "Generate a funny, meme-style caption under 10 words."
    image_part = {"mime_type": "image/jpeg", "data": image_path.read_bytes()}
    response = model.generate_content([prompt, image_part])
    if response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text.strip()
    return "No caption."

def social_share_buttons(text):
    import urllib.parse
    encoded = urllib.parse.quote(text)
    twitter_url = f"https://twitter.com/intent/tweet?text={encoded}"
    whatsapp_url = f"https://api.whatsapp.com/send?text={encoded}"
    linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url={encoded}"
    st.markdown(f"""
    <div style="display:flex; gap:10px; margin:10px 0;">
        <a href="{twitter_url}" target="_blank"><button style="background:#1DA1F2; color:white; border:none; padding:8px 12px; border-radius:8px;">🐦 Twitter</button></a>
        <a href="{whatsapp_url}" target="_blank"><button style="background:#25D366; color:white; border:none; padding:8px 12px; border-radius:8px;">💬 WhatsApp</button></a>
        <a href="{linkedin_url}" target="_blank"><button style="background:#0077B5; color:white; border:none; padding:8px 12px; border-radius:8px;">💼 LinkedIn</button></a>
    </div>
    """, unsafe_allow_html=True)

def get_full_text(results):
    return "\n\n".join([clean_text(r) for r in results])

def get_shareable_text(results):
    text = "# 📜 ClariView Chat Log\n\n"
    for i, description in enumerate(results, 1):
        text += f"**Q/A {i}:**\n{clean_text(description)}\n\n"
    return text

def download_button(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Chat Log</a>'
    return href

# =============================
# 🌐 Social Media Icons (Floating Sidebar)
# =============================
st.markdown("""
<style>
.social-sidebar { position: fixed; top: 40%; left: 20px; display: flex; flex-direction: column; gap: 18px; z-index: 1000; }
.social-icon { width: 42px; height: 42px; border-radius: 50%; display: flex; align-items: center; justify-content: center; background: white; box-shadow: 0 4px 8px rgba(0,0,0,0.2); transition: transform 0.3s ease; }
.social-icon:hover { transform: scale(1.2) rotate(10deg); box-shadow: 0 6px 12px rgba(0,0,0,0.3); }
.social-icon img { width: 24px; height: 24px; }
</style>
<div class="social-sidebar">
    <a class="social-icon" href="https://instagram.com/muthuraj_prince_" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png"/>
    </a>
    <a class="social-icon" href="https://linkedin.com/in/MuthurajC" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/145/145807.png"/>
    </a>
</div>
""", unsafe_allow_html=True)

# =============================
# 🏁 Main App
# =============================
def main():
    # Initialize session state
    if "questions" not in st.session_state:
        st.session_state.questions = ""
    if "results" not in st.session_state:
        st.session_state.results = []
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "mode" not in st.session_state:
        st.session_state.mode = "dark"

    # Sidebar
    st.sidebar.title("🖥️ Navigation")
    page = st.sidebar.radio("Go to", ["Chat: ClariView", "History"])
    st.session_state.mode = st.sidebar.selectbox("🎨 Theme Mode", ["light", "neon", "green-neon", "blue-neon", "dark", "vscode"])

    # Apply theme
    add_custom_css(st.session_state.mode)

    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; text-align: center; padding: 10px; font-size: 14px; color: gray; background-color: #f8f9fa; border-top: 1px solid #ddd;">
        Made with <span style="color:red;">♥</span> by <b>Muthuraj C</b> © 2025
    </div>
    """, unsafe_allow_html=True)

    # Pages
    if page == "Chat: ClariView":
        st.markdown('<div class="animated-title">ClariView - Hacker Image Interpreter</div>', unsafe_allow_html=True)

        uploaded_file = drag_and_drop_upload()
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getvalue())
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            model = initialize_model()
            image_path = Path("temp_image.jpg")

            # Suggestions
            if st.checkbox("💡 Show AI Suggestions"):
                if st.button("🎯 Generate Suggestions"):
                    suggested = suggest_questions(model, image_path)
                    st.session_state.suggested_questions = "\n".join(suggested)
                if "suggested_questions" in st.session_state:
                    st.text_area("Suggested Questions", st.session_state.suggested_questions, height=150)

            # Voice Input
            if st.button("🎙️ Speak Question"):
                spoken = voice_input()
                if spoken:
                    st.session_state.questions += spoken + "\n"

            # Summary Mode
            if st.checkbox("📌 Summary Mode"):
                if st.button("📝 Generate Summary"):
                    summary = generate_summary(model, image_path)
                    hashtags = generate_hashtags(model, summary)
                    st.markdown(f'<div class="chat-bubble">📝 {summary}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-bubble">🏷️ {hashtags}</div>', unsafe_allow_html=True)
                    st.audio(generate_audio(summary, "en"), format="audio/mp3")
                    st.session_state.history.append({
                        "image": uploaded_file,
                        "results": [f"**Summary:** {summary}", f"**Hashtags:** {hashtags}"]
                    })
            else:
                # Q&A Mode
                st.write("💬 Enter your questions:")
                st.session_state.questions = st.text_area("Questions", value=st.session_state.questions, height=120)

                if st.checkbox("🌐 Type in Tamil/Hindi/etc.?"):
                    user_lang_input = st.text_area("Type in your language", "")
                    if user_lang_input:
                        translated, src = translate_to_english(user_lang_input)
                        st.session_state.questions += translated + "\n"
                        st.info(f"Translated ({src} → en): {translated}")

                if st.button("⚡ Generate Description"):
                    questions = [q.strip() for q in st.session_state.questions.split("\n") if q.strip()]
                    if questions:
                        st.session_state.results = generate_content(model, image_path, questions)
                        st.session_state.history.append({
                            "image": uploaded_file,
                            "results": st.session_state.results
                        })
                    else:
                        st.warning("⚠️ Enter at least one question.")

            Path("temp_image.jpg").unlink(missing_ok=True)

            # Display results
            if st.session_state.results:
                for i, desc in enumerate(st.session_state.results):
                    st.markdown(f'<div class="chat-bubble">{desc}</div>', unsafe_allow_html=True)
                    st.audio(generate_audio(desc, "en"), format="audio/mp3")

                    # Meme
                    if st.button("😆 Generate Meme Caption", key=f"meme_{i}"):
                        meme = generate_meme_caption(model, image_path)
                        st.markdown(f'<div class="chat-bubble">😂 {meme}</div>', unsafe_allow_html=True)

                    # Highlight
                    if st.button("🔍 Highlight Faces", key=f"face_{i}"):
                        try:
                            highlighted_img, count = highlight_objects(image_path)
                            st.image(highlighted_img, caption=f"Found {count} face(s)", use_column_width=True)
                        except Exception as e:
                            st.error(f"OpenCV error: {e}")

                    # Batch Translate
                    if st.checkbox("🔊 Generate All Translations", key=f"batch_{i}"):
                        langs = st.multiselect("Languages", ["Tamil", "Hindi", "Malayalam", "Telugu"], default=["Tamil"], key=f"langs_{i}")
                        lang_map = {"Tamil": "ta", "Hindi": "hi", "Malayalam": "ml", "Telugu": "te"}
                        selected_codes = [lang_map[l] for l in langs]
                        if st.button("🔊 Generate All", key=f"gen_{i}"):
                            translations = batch_translate_and_speak(desc, selected_codes)
                            for lang, data in translations.items():
                                st.markdown(f'<div class="chat-bubble">{data["text"]}</div>', unsafe_allow_html=True)
                                st.audio(data["audio"], format="audio/mp3")

                    # Share
                    full_text = clean_text(desc)
                    social_share_buttons(full_text)

                # QR Code
                if st.button("📱 Generate QR Code"):
                    qr_path = generate_qr_code(get_full_text(st.session_state.results))
                    st.image(qr_path, caption="Scan to share")

        else:
            st.info("📤 Upload an image to begin")

    elif page == "History":
        st.markdown('<div class="animated-title">📜 Hack Logs - History</div>', unsafe_allow_html=True)
        if st.session_state.history:
            query = st.text_input("🔍 Search in history")
            filtered = search_history(query) if query else enumerate(st.session_state.history)

            for idx, entry in filtered:
                st.write(f"**Log {idx+1}**")
                st.image(entry["image"], caption=f"Image {idx+1}", use_column_width=True)
                for desc in entry["results"]:
                    st.markdown(f'<div class="chat-bubble">{desc}</div>', unsafe_allow_html=True)

                share_text = get_shareable_text(entry["results"])
                st.code(share_text, language="markdown")
                st.markdown(download_button(share_text, f"ClariView_Log_{idx+1}.txt"), unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📤 Export as PDF", key=f"pdf_{idx}"):
                        export_to_pdf(share_text)
                        with open("ClariView_Log.pdf", "rb") as f:
                            st.download_button("⬇️ Download PDF", f, "ClariView_Log.pdf")
                with col2:
                    if st.button("📄 Export as Word", key=f"docx_{idx}"):
                        export_to_word(share_text)
                        with open("ClariView_Log.docx", "rb") as f:
                            st.download_button("⬇️ Download Word", f, "ClariView_Log.docx")

                st.markdown("---")
        else:
            st.info("ℹ️ No logs yet.")

if __name__ == "__main__":
    main()
