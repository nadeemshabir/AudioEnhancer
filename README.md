# AudioEnhancer
🗣️ AudioEnhancer — AI Voice-Over & Video Enhancement Tool
🎬 Overview

AudioEnhancer is an AI-powered tool that automatically:

Generates voiceovers using cloned or custom voices,

Adds subtitles and aligns them perfectly with video content,

Enhances the audio quality and synchronizes speech using FFmpeg,

Integrates with Google Sheets for tracking and AWS Polly / Deepgram / Gemini / S3Gen voice models for text-to-speech and transcription.

This project helps automate multilingual video dubbing and content creation at scale.

🚀 Features

🎤 Voice Cloning: Generate natural-sounding voiceovers in multiple languages (supports English, German, Spanish, etc.)

🧠 AI-Powered Syncing: Auto-aligns subtitles with speech using timestamps

🔊 Audio Enhancement: Uses FFmpeg for clear, noise-free mixing

🪄 Cloud Integration: AWS Polly for TTS, Deepgram for STT, Google Sheets for data automation

💬 Multilingual Support: Detects and converts speech from different languages

⚡ Batch Processing: Supports multiple videos and updates progress automatically

🧩 Project Structure
AudioEnhancer/
│
├── main.py                   # Main FastAPI backend
├── requirements.txt           # Python dependencies
├── Data/
│   ├── tmp/                   # Temporary files (auto-generated)
│   ├── Original_videos/       # Uploaded input videos
│   └── Final_videos/          # Processed output videos
├── voice_clone/               # Custom voice cloning module
│   └── src/chatterbox/        # Voice model implementations
└── .gitignore

⚙️ Installation

Clone the repository:

git clone https://github.com/nadeemshabir/AudioEnhancer.git
cd AudioEnhancer


Create a virtual environment:

python -m venv venv
venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Set up your environment variables:

AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
DEEPGRAM_API_KEY=your_key
GOOGLE_APPLICATION_CREDENTIALS=service_account.json

▶️ Usage

Run the FastAPI server:

python main.py


Upload videos through the frontend or API endpoint:

POST /process-video


Monitor progress logs in the console.
Final videos with AI-generated voiceovers will appear in:

Data/Final_videos/

🧠 Technologies Used
Type	Tools
Backend	FastAPI, Uvicorn
AI Models	AWS Polly, Deepgram, Gemini, S3Gen Voice Cloner
Video Processing	FFmpeg
Data	Google Sheets API
Environment	Python 3.11+
🧾 Example Workflow

Upload reference voice (can be in any language)

Upload video for enhancement

The system clones tone/style, transcribes, generates subtitles, and syncs the final video

The output video is saved with aligned voice + captions
