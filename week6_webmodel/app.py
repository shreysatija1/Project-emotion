"""
Production Web Application for Speech Emotion Recognition (SER).
FastAPI backend with embedded browser microphone recording UI.
Endpoints:
- GET / : Interactive Web Audio recorder & emotion predictor interface
- GET /health : System & model health check
- POST /predict : Audio emotion classification endpoint (file or base64)
"""

import base64
import time
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
WEEK6_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEEK6_DIR))

from inference import predict_emotion, EMOTION_LABELS

app = FastAPI(
    title="Speech Emotion Recognition Web API",
    description="Leak-free, deployable 6-class Speech Emotion Recognition for web microphones.",
    version="1.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Base64PredictRequest(BaseModel):
    audio_base64: str


@app.get("/health")
def health_check():
    """Health status and model capability check."""
    return {
        "status": "ok",
        "service": "Speech Emotion Recognition Web API",
        "version": "1.0.0",
        "classes": EMOTION_LABELS,
        "sample_rate": 16000,
        "deployability": "per-utterance leak-free"
    }


@app.post("/predict")
async def predict_endpoint(
    file: Optional[UploadFile] = File(None),
    payload: Optional[Base64PredictRequest] = Body(None)
):
    """
    Accepts an audio file (.wav, .flac, .mp3, .ogg, .webm) or base64 payload.
    Returns 6 calibrated emotion probabilities, top prediction, and uncertainty flags.
    """
    audio_bytes = None
    
    if file is not None:
        audio_bytes = await file.read()
    elif payload is not None and payload.audio_base64:
        try:
            # Strip data URL header if present (e.g. data:audio/wav;base64,...)
            raw_b64 = payload.audio_base64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio payload: {e}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded audio file or audio_base64.")
        
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Received empty audio payload.")
        
    try:
        result = predict_emotion(audio_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Embedded, self-contained Web Audio microphone recording prototype."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Speech Emotion Recognition Prototype</title>
        <style>
            :root {
                --bg: #0f172a;
                --card: #1e293b;
                --border: #334155;
                --text: #f8fafc;
                --muted: #94a3b8;
                --primary: #38bdf8;
                --danger: #f43f5e;
                --success: #10b981;
                --warning: #f59e0b;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 2rem 1rem;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .container {
                max-width: 640px;
                width: 100%;
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
            }
            h1 {
                margin-top: 0;
                font-size: 1.5rem;
                text-align: center;
                color: var(--primary);
            }
            p.subtitle {
                text-align: center;
                color: var(--muted);
                font-size: 0.9rem;
                margin-bottom: 2rem;
            }
            .controls {
                display: flex;
                gap: 1rem;
                justify-content: center;
                margin-bottom: 1.5rem;
            }
            button {
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                border: none;
                transition: all 0.2s;
            }
            #recordBtn {
                background-color: var(--danger);
                color: white;
            }
            #recordBtn.recording {
                background-color: #e11d48;
                box-shadow: 0 0 15px rgba(244, 63, 94, 0.6);
            }
            #uploadBtn {
                background-color: var(--border);
                color: var(--text);
            }
            .status-indicator {
                text-align: center;
                font-size: 0.85rem;
                color: var(--muted);
                margin-bottom: 1.5rem;
                min-height: 1.2rem;
            }
            .badge-low-conf {
                display: inline-block;
                background-color: rgba(245, 158, 11, 0.2);
                color: var(--warning);
                border: 1px solid var(--warning);
                padding: 0.25rem 0.6rem;
                border-radius: 4px;
                font-size: 0.75rem;
                margin-top: 0.5rem;
            }
            .bars-container {
                display: flex;
                flex-direction: column;
                gap: 0.8rem;
                margin-top: 1.5rem;
            }
            .bar-row {
                display: flex;
                flex-direction: column;
                gap: 0.2rem;
            }
            .bar-header {
                display: flex;
                justify-content: space-between;
                font-size: 0.85rem;
                text-transform: capitalize;
            }
            .bar-track {
                width: 100%;
                height: 10px;
                background-color: #0f172a;
                border-radius: 5px;
                overflow: hidden;
            }
            .bar-fill {
                height: 100%;
                background-color: var(--primary);
                width: 0%;
                transition: width 0.4s ease;
                border-radius: 5px;
            }
            .bar-fill.highlight {
                background-color: var(--success);
            }
            .stats-meta {
                margin-top: 1.5rem;
                padding-top: 1rem;
                border-top: 1px solid var(--border);
                display: flex;
                justify-content: space-between;
                font-size: 0.75rem;
                color: var(--muted);
            }
            input[type="file"] {
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Speech Emotion Recognition</h1>
            <p class="subtitle">Real-Time Acoustic Prediction for Browser Audio</p>
            
            <div class="controls">
                <button id="recordBtn">Record Voice (Mic)</button>
                <button id="uploadBtn">Upload Audio File</button>
                <input type="file" id="fileInput" accept="audio/*">
            </div>
            
            <div id="statusText" class="status-indicator">Click "Record Voice" to speak or upload a recording.</div>
            <div id="badgeContainer" style="text-align: center;"></div>

            <div class="bars-container" id="barsList"></div>

            <div class="stats-meta">
                <span id="latencySpan">Latency: -- ms</span>
                <span id="durationSpan">Duration: -- s</span>
                <span id="entropySpan">Entropy: --</span>
            </div>
        </div>

        <script>
            const EMOTIONS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"];
            const barsList = document.getElementById("barsList");
            const recordBtn = document.getElementById("recordBtn");
            const uploadBtn = document.getElementById("uploadBtn");
            const fileInput = document.getElementById("fileInput");
            const statusText = document.getElementById("statusText");
            const badgeContainer = document.getElementById("badgeContainer");

            let mediaRecorder = null;
            let audioChunks = [];

            // Initialize probability bars
            EMOTIONS.forEach(emo => {
                const row = document.createElement("div");
                row.className = "bar-row";
                row.innerHTML = `
                    <div class="bar-header">
                        <span>${emo}</span>
                        <span id="val-${emo}">0.0%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" id="fill-${emo}"></div>
                    </div>
                `;
                barsList.appendChild(row);
            });

            uploadBtn.addEventListener("click", () => fileInput.click());
            fileInput.addEventListener("change", async (e) => {
                if (!e.target.files.length) return;
                const file = e.target.files[0];
                statusText.innerText = "Analyzing file: " + file.name + "...";
                const formData = new FormData();
                formData.append("file", file);
                await sendAudio(formData);
            });

            recordBtn.addEventListener("click", async () => {
                if (!mediaRecorder || mediaRecorder.state === "inactive") {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        mediaRecorder = new MediaRecorder(stream);
                        audioChunks = [];
                        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
                        mediaRecorder.onstop = async () => {
                            const blob = new Blob(audioChunks, { type: "audio/webm" });
                            const formData = new FormData();
                            formData.append("file", blob, "recording.webm");
                            statusText.innerText = "Classifying speech recording...";
                            await sendAudio(formData);
                            stream.getTracks().forEach(t => t.stop());
                        };
                        mediaRecorder.start();
                        recordBtn.innerText = "Stop Recording";
                        recordBtn.classList.add("recording");
                        statusText.innerText = "Listening... Speak naturally (1-5 seconds).";
                    } catch (err) {
                        statusText.innerText = "Microphone access denied: " + err.message;
                    }
                } else {
                    mediaRecorder.stop();
                    recordBtn.innerText = "Record Voice (Mic)";
                    recordBtn.classList.remove("recording");
                }
            });

            async function sendAudio(formData) {
                try {
                    const res = await fetch("/predict", { method: "POST", body: formData });
                    if (!res.ok) {
                        const err = await res.json();
                        statusText.innerText = "Error: " + (err.detail || "Server error");
                        return;
                    }
                    const data = await res.json();
                    renderResult(data);
                } catch (err) {
                    statusText.innerText = "Network error: " + err.message;
                }
            }

            function renderResult(data) {
                statusText.innerText = `Prediction: ${data.predicted_emotion.toUpperCase()} (${(data.confidence * 100).toFixed(1)}% confidence)`;
                
                badgeContainer.innerHTML = "";
                if (data.low_confidence) {
                    badgeContainer.innerHTML = `<span class="badge-low-conf">&#9888; Low Confidence: ${data.low_confidence_reason || "Uncertain recording"}</span>`;
                }

                EMOTIONS.forEach(emo => {
                    const prob = data.probabilities[emo] || 0.0;
                    const fill = document.getElementById(`fill-${emo}`);
                    const val = document.getElementById(`val-${emo}`);
                    const pct = (prob * 100).toFixed(1) + "%";
                    val.innerText = pct;
                    fill.style.width = pct;
                    if (emo === data.predicted_emotion && !data.low_confidence) {
                        fill.classList.add("highlight");
                    } else {
                        fill.classList.remove("highlight");
                    }
                });

                document.getElementById("latencySpan").innerText = `Latency: ${data.inference_latency_ms} ms`;
                document.getElementById("durationSpan").innerText = `Duration: ${data.duration_seconds} s`;
                document.getElementById("entropySpan").innerText = `Entropy: ${data.entropy}`;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
