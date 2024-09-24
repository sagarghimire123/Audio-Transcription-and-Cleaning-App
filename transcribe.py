import streamlit as st
import openai
import os
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(file):
    """Transcribe audio file using OpenAI's Whisper model."""
    audio_file = BytesIO(file.read())
    
    # Perform transcription
    response = openai.Audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    
    return response['text']

def clean_transcript(transcript):
    """Clean the transcript using GPT-3.5-turbo."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You will be provided a transcript which is messy. Clean it up and give me a polished, well-written English transcript. Remove any filler words."},
            {"role": "user", "content": transcript}
        ]
    )
    
    return response.choices[0].message['content']

# Streamlit interface
st.title("Audio Transcript Cleaner")

st.write("Upload an audio file (e.g., m4a format) to clean the transcript:")

uploaded_file = st.file_uploader("Choose an audio file", type=["m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/m4a')

    if st.button('Clean Transcript'):
        with st.spinner('Processing...'):
            # Transcribe the audio file
            raw_transcript = transcribe_audio(uploaded_file)
            
            # Clean the transcript
            cleaned_transcript = clean_transcript(raw_transcript)
            
            st.subheader("Cleaned Transcript:")
            st.write(cleaned_transcript)
