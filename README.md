# 🗣️ AudioEnhancer — AI Voice-Over & Video Enhancement Tool

## 🎬 Overview
**AudioEnhancer** is an AI-powered system that automates:
- Generating **voiceovers** using cloned or reference voices  
- Creating and aligning **subtitles** automatically  
- Enhancing and synchronizing **audio and video** using FFmpeg  
- Optionally integrating with **AWS**, **Deepgram**, and **Gemini AI** for speech and text processing  

This project can run **fully offline on your local machine**, or optionally connect to the cloud for extended capabilities.

---

## 💻 Run Locally — No Cloud Required

### ✅ Local Mode Highlights
- Works entirely on your computer — no API keys or internet needed  
- Uses **local TTS (voice cloning)** and **FFmpeg** for processing  
- Ideal for testing, development, or private offline workflows  

Just clone, set up dependencies, and run.  

---

## ☁️ Optional Cloud Mode

If you want more power — like high-quality transcription, Gemini text refinement, or S3 storage — you can enable:
- **AWS S3 / Polly** for TTS and storage  
- **Deepgram** for transcription  
- **Google Sheets API** for cloud-based tracking  

Simply set up your `.env` (instructions below).

---

## 🚀 Features
- 🎤 **Voice Cloning:** Generate natural speech using your own or reference voices  
- 🧠 **AI Text Refinement:** Optionally uses Gemini AI to clean up transcripts  
- 🔊 **Audio Enhancement:** Auto-syncs audio with video and adds subtitles  
- 💬 **Multilingual Support:** Works with English, German, Spanish, and more  
- ⚙️ **Two Modes:** Run **locally** or **via AWS/Deepgram** with environment setup  
- 📊 **Google Sheets Sync (Optional):** Automatically logs timings and updates sheets  

---

## 🧩 Project Structure
AudioEnhancer/
│
├── main.py # Main FastAPI app for processing
├── requirements.txt # Dependencies
├── Data/
│ ├── tmp/ # Temporary files (auto-generated)
│ ├── Original_videos/ # Input videos
│ └── Final_videos/ # Final outputs
├── voice_clone/ # Voice cloning logic
│ └── src/chatterbox/ # Internal TTS and voice modules
└── .gitignore


---

## ⚙️ Installation

### 1️⃣ Clone this repo
```bash
git clone https://github.com/nadeemshabir/AudioEnhancer.git
cd AudioEnhancer
```
###2️⃣ Create and activate virtual environment
```
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate   # Mac/Linux
```
###3️⃣ Install dependencies
```
pip install -r requirements.txt
```
🧩 Configuration
▶️ To run locally (default mode)

No cloud keys are needed.
The app will use your local:

Voice cloning models from voice_clone/

FFmpeg for audio/video handling

Ref voice from Ref_voice/

Just make sure you have:

FFmpeg installed and added to PATH

A reference voice file in Ref_voice/Anshul_Ref_Voice_trimmed.wav

