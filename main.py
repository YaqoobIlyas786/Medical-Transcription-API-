import os
import json
import logging
import aiohttp
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# ─── Logging Setup ────────────────────────────────────────────────────────────────
import sys

# Configure logging for both local and cloud deployment
log_level = logging.DEBUG if os.getenv("ENVIRONMENT") != "production" else logging.INFO

# Configure logging to work with cloud platforms
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Ensure logs go to stdout for cloud platforms
    ]
)

# Set specific loggers
logger = logging.getLogger("server")
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(log_level)

# Suppress verbose third-party library logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("aiofiles").setLevel(logging.WARNING)

# Ensure aiohttp and other libraries don't flood logs in production
if os.getenv("ENVIRONMENT") == "production":
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("python_multipart.multipart").setLevel(logging.WARNING)
else:
    logging.getLogger("aiohttp").setLevel(logging.INFO)
    # Suppress verbose multipart logs even in development
    logging.getLogger("python_multipart.multipart").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)

# Completely disable multipart debug logs by setting to CRITICAL
logging.getLogger("python_multipart.multipart").setLevel(logging.CRITICAL)

# ─── Load API Keys ────────────────────────────────────────────────────────────────
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DEEPGRAM_API_KEY or not OPENAI_API_KEY:
    raise RuntimeError("API keys not set in .env")
    
logger.info("Loaded Deepgram key and OpenAI key")

# ─── FastAPI Setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical Transcription API", 
    description="API for transcribing audio and generating SOAP notes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure file upload limits and timeouts
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio

class FileUploadMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/transcribe_audio" and request.method == "POST":
            upload_start = time.time()
            logger.info(f"🔄 Upload middleware: Request started at {time.strftime('%H:%M:%S.%f')[:-3]}")
            logger.info(f"📊 Content-Length: {request.headers.get('content-length', 'unknown')}")
            logger.info(f"🌐 Content-Type: {request.headers.get('content-type', 'unknown')}")
        
        response = await call_next(request)
        
        if request.url.path == "/transcribe_audio" and request.method == "POST":
            upload_end = time.time()
            total_middleware_time = upload_end - upload_start
            logger.info(f"✅ Upload middleware: Total request time {total_middleware_time:.2f}s")
        
        return response

app.add_middleware(FileUploadMiddleware)

# CORS middleware to allow frontend access from localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ─── Deepgram API URL ────────────────────────────────────────────────────────────
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true&diarize=true&smart_format=true&interim_results=false&utterances=true"

# ─── OpenAI API URL ──────────────────────────────────────────────────────────────
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# ─── Response Models ──────────────────────────────────────────────────────────────
class TranscriptionResponse(BaseModel):
    transcription: str

class ClassificationResponse(BaseModel):
    classification: str

class SpeakerClassificationResponse(BaseModel):
    classified_transcription: str

class SOAPNotesResponse(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str

class TranscriptionRequest(BaseModel):
    transcription: str

# ─── Root Endpoint ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """
    Root endpoint providing API information and available endpoints.
    """
    return {
        "message": "Medical Transcription API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "POST /transcribe_audio": "Upload audio file for transcription (auto-optimized for production)",
            "POST /classify_speakers": "Classify speakers as Doctor/Patient in transcription",
            "POST /classify_or_summarize": "Get clinical summary and classification",
            "POST /generate_soap": "Generate SOAP notes from transcription"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }

# ─── Health Check Endpoint ─────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "timestamp": "2025-06-25",
        "api_version": "1.0.0"
    }

# ─── Logging Test Endpoint ─────────────────────────────────────────────────────
@app.get("/test-logs")
async def test_logs():
    """
    Test endpoint to verify logging is working in production.
    """
    logger.debug("DEBUG: This is a debug message")
    logger.info("INFO: This is an info message")
    logger.warning("WARNING: This is a warning message")
    logger.error("ERROR: This is an error message")
    
    return {
        "status": "logs_tested",
        "message": "Check your deployment logs to see if these messages appear",
        "timestamp": "2025-06-26",
        "environment": os.getenv("ENVIRONMENT", "local")
    }

# ─── Transcribe Audio Using Deepgram ───────────────────────────────────────────────
@app.post("/transcribe_audio", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint to receive audio file and perform transcription via Deepgram with speaker diarization.
    Optimized for maximum speed in production with detailed logging in development.
    """
    import time
    
    # Check if we're in production mode for speed optimization
    is_production = os.getenv("ENVIRONMENT") == "production"
    
    if not is_production:
        # Development mode - detailed logging
        start_time = time.time()
        logger.info("=== AUDIO PROCESSING STARTED ===")
        logger.info(f"🚀 Processing file: {file.filename}")
        logger.info(f"📋 Content type: {file.content_type}")
    
    # Determine audio format quickly
    content_type = "audio/wav"  # Default
    if file.filename:
        if file.filename.endswith('.webm'):
            content_type = "audio/webm"
        elif file.filename.endswith('.mp4') or file.filename.endswith('.m4a'):
            content_type = "audio/mp4"
        elif file.filename.endswith('.ogg'):
            content_type = "audio/ogg"
        elif file.filename.endswith('.flac'):
            content_type = "audio/flac"
    
    if not is_production:
        logger.info(f"🔧 Using content type: {content_type}")
    
    try:
        # Production optimized streaming
        if not is_production:
            logger.info("📡 Streaming to Deepgram...")
            stream_start = time.time()
        
        timeout = aiohttp.ClientTimeout(total=180 if is_production else 300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": content_type
            }
            
            # Ultra-fast streaming for production, detailed tracking for development
            if is_production:
                # Production: Maximum speed, minimal logging
                async def file_stream():
                    while chunk := await file.read(65536):  # 16KB chunks for speed
                        yield chunk
            else:
                # Development: Progress tracking
                async def file_stream():
                    total_bytes = 0
                    chunk_count = 0
                    while True:
                        chunk = await file.read(8192)  # 8KB chunks
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        chunk_count += 1
                        
                        # Log every 100 chunks (approximately every 800KB)
                        if chunk_count % 100 == 0:
                            logger.info(f"📤 Streamed {total_bytes/(1024*1024):.1f}MB...")
                        
                        yield chunk
                    
                    logger.info(f"✅ Stream complete: {total_bytes/(1024*1024):.2f}MB in {chunk_count} chunks")
            
            # Stream to Deepgram
            async with session.post(DEEPGRAM_API_URL, headers=headers, data=file_stream()) as response:
                if response.status != 200:
                    error_text = await response.text()
                    if not is_production:
                        logger.error(f"❌ Deepgram API error: {response.status}, {error_text}")
                    raise HTTPException(status_code=500, detail=f"Error transcribing audio: {error_text}")
                
                data = await response.json()
                
                if not is_production:
                    stream_end = time.time()
                    stream_time = stream_end - stream_start
                    total_time = stream_end - start_time
                    logger.info(f"⚡ Deepgram response received in {stream_time:.2f}s")
                    logger.info(f"🏁 Total processing time: {total_time:.2f}s")
                
                # Quick transcription extraction (optimized for both modes)
                utterances = data.get("results", {}).get("utterances", [])
                if utterances:
                    if not is_production:
                        logger.info(f"🎯 Found {len(utterances)} utterances")
                    
                    # Fast list comprehension instead of loop
                    formatted_transcription = "\n".join([
                        f"Speaker {u.get('speaker', 0)}: {u.get('transcript', '').strip()}"
                        for u in utterances if u.get('transcript', '').strip()
                    ])
                else:
                    # Fallback to regular transcription
                    channels = data.get("results", {}).get("channels", [])
                    if channels:
                        transcript = channels[0].get("alternatives", [{}])[0].get("transcript", "")
                        formatted_transcription = f"Speaker 0: {transcript}"
                    else:
                        raise HTTPException(status_code=500, detail="No transcription channels found")
                    
                    if not is_production:
                        logger.warning("⚠️ No utterances found, using fallback")

                if not formatted_transcription.strip():
                    raise HTTPException(status_code=500, detail="Failed to get transcription from Deepgram")
                
                if not is_production:
                    logger.info(f"✨ Transcription success: {len(formatted_transcription)} characters")
                    logger.info(f"=== END AUDIO PROCESSING ===")
                
                return TranscriptionResponse(transcription=formatted_transcription.strip())
                
    except Exception as e:
        if not is_production:
            logger.error(f"❌ Error in transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# ─── Classify Speakers as Doctor/Patient ─────────────────────────────────────────────
@app.post("/classify_speakers", response_model=SpeakerClassificationResponse)
async def classify_speakers(request: TranscriptionRequest):
    """
    Classify speakers in the transcription as Doctor or Patient using OpenAI.
    """
    transcription = request.transcription
    logger.info(f"Classifying speakers for transcription of length: {len(transcription)}")
    
    # For very long transcriptions, we need to increase max_tokens
    # Estimate required tokens (rough approximation: 1 token ≈ 4 characters)
    estimated_tokens = len(transcription) // 4
    max_tokens = min(max(estimated_tokens + 100, 1000), 4000)  # Between 1000-4000 tokens
    
    logger.info(f"Using max_tokens: {max_tokens} for classification")
    
    openai_payload = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are a medical AI assistant. Analyze the following medical consultation transcription with speaker labels (Speaker 0, Speaker 1, etc.) and classify each speaker as either 'Doctor' or 'Patient'. Return the COMPLETE transcription with speakers relabeled as 'Doctor:' or 'Patient:' based on who is speaking. Doctors typically ask questions, provide medical advice, and use medical terminology. Patients typically describe symptoms and answer questions. IMPORTANT: Return the full transcription - do not truncate or summarize."
            },
            {
                "role": "user",
                "content": f"Please classify the speakers in this medical transcription and return the COMPLETE transcription with Doctor/Patient labels (do not truncate):\n\n{transcription}"
            }        ],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(OPENAI_API_URL, headers=headers, json=openai_payload) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Error with OpenAI API response: {response.status}, {error_text}")
                raise HTTPException(status_code=500, detail="Error classifying speakers")

            data = await response.json()
            classified_transcription = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            logger.info(f"Classified transcription length: {len(classified_transcription)}")
            logger.info(f"Original vs classified ratio: {len(classified_transcription)}/{len(transcription)} = {len(classified_transcription)/len(transcription):.2f}")

            if not classified_transcription:
                logger.error("OpenAI returned empty classified transcription")
                raise HTTPException(status_code=500, detail="Failed to classify speakers")

            # Check if transcription was truncated (less than 80% of original)
            if len(classified_transcription) < len(transcription) * 0.8:
                logger.warning(f"Classified transcription appears truncated: {len(classified_transcription)} vs {len(transcription)}")

            return SpeakerClassificationResponse(classified_transcription=classified_transcription)

# ─── Send Transcription to OpenAI for Classification or Summarization ──────────────
@app.post("/classify_or_summarize", response_model=ClassificationResponse)
async def classify_or_summarize(request: TranscriptionRequest):
    """
    Classify or summarize the transcription using OpenAI models.
    """
    transcription = request.transcription
    
    openai_payload = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are a medical AI assistant that analyzes medical consultation transcriptions. Provide a brief clinical summary and classification of the consultation type (e.g., routine check-up, follow-up visit, emergency consultation, prenatal care, etc.). Include key medical points discussed and any immediate concerns."
            },
            {
                "role": "user",
                "content": f"Analyze this medical consultation transcription and provide a clinical classification and summary:\n\n{transcription}"
            }
        ],
        "max_tokens": 200,
        "temperature": 0.3
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(OPENAI_API_URL, headers=headers, json=openai_payload) as response:
            if response.status != 200:
                logger.error(f"Error with OpenAI API response: {response.status}")
                raise HTTPException(status_code=500, detail="Error with classification or summarization")

            data = await response.json()
            classification = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if not classification:
                raise HTTPException(status_code=500, detail="Failed to classify or summarize text")

            return ClassificationResponse(classification=classification)

# ─── Generate SOAP Notes Using OpenAI ─────────────────────────────────────────────
@app.post("/generate_soap", response_model=SOAPNotesResponse)
async def generate_soap(request: TranscriptionRequest):
    """
    Generate SOAP notes from the transcription using OpenAI models.
    """
    transcription = request.transcription
    openai_payload = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates SOAP notes from medical transcriptions. Please analyze the following transcription and generate the SOAP notes in the format: Subjective: [content] Objective: [content] Assessment: [content] Plan: [content]"
            },
            {
                "role": "user",
                "content": transcription
            }
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(OPENAI_API_URL, headers=headers, json=openai_payload) as response:
            if response.status != 200:
                logger.error(f"Error with OpenAI API response: {response.status}")
                raise HTTPException(status_code=500, detail="Error generating SOAP notes")

            data = await response.json()
            soap_notes = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if not soap_notes:
                raise HTTPException(status_code=500, detail="Failed to generate SOAP notes")

            # Parse the SOAP notes into separate components
            try:
                lines = soap_notes.split("\n")
                subjective = objective = assessment = plan = ""
                current_section = ""
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("Subjective:"):
                        current_section = "subjective"
                        subjective = line.replace("Subjective:", "").strip()
                    elif line.startswith("Objective:"):
                        current_section = "objective"
                        objective = line.replace("Objective:", "").strip()
                    elif line.startswith("Assessment:"):
                        current_section = "assessment"
                        assessment = line.replace("Assessment:", "").strip()
                    elif line.startswith("Plan:"):
                        current_section = "plan"
                        plan = line.replace("Plan:", "").strip()
                    elif line and current_section:
                        # Continue adding to the current section
                        if current_section == "subjective":
                            subjective += " " + line
                        elif current_section == "objective":
                            objective += " " + line
                        elif current_section == "assessment":
                            assessment += " " + line
                        elif current_section == "plan":
                            plan += " " + line
                
                # Clean up the sections
                subjective = subjective.strip()
                objective = objective.strip()
                assessment = assessment.strip()
                plan = plan.strip()
                
                # If parsing fails, provide default structure
                if not subjective and not objective and not assessment and not plan:
                    subjective = soap_notes
                    objective = "Physical examination and vital signs to be documented"
                    assessment = "Medical assessment based on patient presentation"
                    plan = "Treatment plan to be determined"
                        
            except Exception as e:
                logger.error(f"Error parsing SOAP notes: {e}")
                raise HTTPException(status_code=500, detail="Error parsing SOAP notes")

            return SOAPNotesResponse(subjective=subjective, objective=objective, assessment=assessment, plan=plan)

# ─── Server Startup ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (for deployment) or default to 8080
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "127.0.0.1")
    
    logger.info("Starting Medical Transcription API server...")
    logger.info(f"Server will run on {host}:{port}")
    logger.info(f"API Documentation available at: http://{host}:{port}/docs")
    logger.info(f"Alternative docs at: http://{host}:{port}/redoc")
    logger.info(f"Health check at: http://{host}:{port}/health")
    
    # Run the FastAPI server with deployment-friendly configuration
    uvicorn.run(
        "main:app", 
        host=host, 
        port=port, 
        log_level="debug" if os.getenv("ENVIRONMENT") != "production" else "info",
        reload=False,  # Disable reload for production
        access_log=True,
        use_colors=False,  # Disable colors for cloud logs
        loop="asyncio",  # Specify event loop for better cloud compatibility
        # Optimize for file uploads
        limit_max_requests=1000,
        backlog=2048,
        # Increase timeout for large file uploads
        timeout_keep_alive=65,
        h11_max_incomplete_event_size=16777216,  # 16MB for large uploads
    )
