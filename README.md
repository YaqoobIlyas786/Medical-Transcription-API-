# Medical Transcription API

A FastAPI-based medical transcription service that converts audio recordings into text with speaker diarization and generates SOAP notes.

## Features

- 🎤 **Audio Transcription**: Convert medical audio recordings to text using Deepgram
- 👥 **Speaker Diarization**: Identify and separate different speakers
- 🏥 **Doctor/Patient Classification**: Automatically classify speakers as doctor or patient
- 📋 **SOAP Notes Generation**: Generate structured medical notes using OpenAI
- 📊 **Clinical Summarization**: Provide medical consultation summaries

## API Endpoints

- `GET /` - API information and status
- `GET /health` - Health check endpoint
- `POST /transcribe_audio` - Upload audio file for transcription
- `POST /classify_speakers` - Classify speakers as Doctor/Patient
- `POST /classify_or_summarize` - Get clinical summary
- `POST /generate_soap` - Generate SOAP notes

## Environment Variables

```
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Deployment

This application is configured for deployment on Render.com

## Documentation

- API Documentation: `/docs`
- Alternative Documentation: `/redoc`
