from __future__ import annotations

import os
import re
import shutil
import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from qwen_asr import Qwen3ASRModel



load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

MODEL_PATH = os.getenv("ASR_STREAM_MODEL", os.getenv("ASR_MODEL", "/home/regchen/Chuyi/models/Qwen3-ASR-0.6B"))
GPU_MEMORY_UTILIZATION = float(os.getenv("ASR_STREAM_GPU_MEMORY_UTILIZATION", "0.08"))
UNFIXED_CHUNK_NUM = int(os.getenv("ASR_STREAM_UNFIXED_CHUNK_NUM", "4"))
UNFIXED_TOKEN_NUM = int(os.getenv("ASR_STREAM_UNFIXED_TOKEN_NUM", "5"))
CHUNK_SIZE_SEC = float(os.getenv("ASR_STREAM_CHUNK_SIZE_SEC", "1.0"))
MAX_MODEL_LEN = int(os.getenv("ASR_STREAM_MAX_MODEL_LEN", "4096"))
MAX_NUM_SEQS = int(os.getenv("ASR_STREAM_MAX_NUM_SEQS", "1"))
MAX_NUM_BATCHED_TOKENS = int(os.getenv("ASR_STREAM_MAX_NUM_BATCHED_TOKENS", "1024"))
TMP_DIR = Path(os.getenv("ASR_STREAM_TMP_DIR", "/tmp/meeting-ai-asr"))
SAMPLE_RATE = int(os.getenv("ASR_STREAM_SAMPLE_RATE", "16000"))
SEGMENT_MAX_CHARS = int(os.getenv("ASR_STREAM_SEGMENT_MAX_CHARS", "42"))
START_TIMEOUT_SEC = float(os.getenv("ASR_STREAM_START_TIMEOUT_SEC", "20"))
CHUNK_TIMEOUT_SEC = float(os.getenv("ASR_STREAM_CHUNK_TIMEOUT_SEC", "45"))
FINISH_TIMEOUT_SEC = float(os.getenv("ASR_STREAM_FINISH_TIMEOUT_SEC", "45"))


app = FastAPI(title="Qwen ASR Low Latency Streaming", version="1.0.0")
asr: Qwen3ASRModel | None = None
asr_lock = threading.Lock()
sessions_lock = threading.Lock()


@dataclass
class Session:
    state: Any
    created_at: float
    last_seen: float
    audio_seconds: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)


SESSIONS: dict[str, Session] = {}
SESSION_TTL_SEC = 10 * 60


def _gc_sessions() -> None:
    if asr is None:
        return
    now = time.time()
    with sessions_lock:
        dead = [sid for sid, item in SESSIONS.items() if now - item.last_seen > SESSION_TTL_SEC]
        for sid in dead:
            SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Session:
    _gc_sessions()
    with sessions_lock:
        session = SESSIONS.get(session_id)
        if not session:
            raise HTTPException(status_code=409, detail="session expired or replaced")
        session.last_seen = time.time()
        return session


def _error_detail(exc: Exception, fallback: str) -> str:
    return str(exc).strip() or fallback


async def _run_asr_locked(func: Any, *args: Any, timeout: float, **kwargs: Any) -> Any:
    def call() -> Any:
        with asr_lock:
            return func(*args, **kwargs)

    return await asyncio.wait_for(asyncio.to_thread(call), timeout=timeout)


def _clean_stream_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_long_piece(piece: str) -> list[str]:
    if len(piece) <= SEGMENT_MAX_CHARS:
        return [piece]

    parts: list[str] = []
    current = ""
    for char in piece:
        current += char
        if len(current) >= SEGMENT_MAX_CHARS and char in "，,、 ":
            parts.append(current.strip())
            current = ""
    if current.strip():
        parts.append(current.strip())

    if any(len(part) > SEGMENT_MAX_CHARS * 1.5 for part in parts) or not parts:
        parts = [piece[index : index + SEGMENT_MAX_CHARS].strip() for index in range(0, len(piece), SEGMENT_MAX_CHARS)]
    return [part for part in parts if part]


def _split_stream_text(text: str) -> list[str]:
    cleaned = _clean_stream_text(text)
    if not cleaned:
        return []

    pieces = [match.group(0).strip() for match in re.finditer(r"[^。！？!?；;\n]+[。！？!?；;]?", cleaned)]
    if not pieces:
        pieces = [cleaned]

    segments: list[str] = []
    for piece in pieces:
        segments.extend(_split_long_piece(piece))
    return [segment for segment in segments if segment]


def _segments_from_text(session_id: str, text: str, total_seconds: float, language: str) -> list[dict[str, Any]]:
    parts = _split_stream_text(text)
    if not parts:
        return []

    total_seconds = max(0.0, total_seconds)
    weights = [max(1, len(part)) for part in parts]
    total_weight = max(1, sum(weights))
    cursor = 0.0
    output: list[dict[str, Any]] = []
    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            end = total_seconds
        else:
            end = total_seconds * sum(weights[: index + 1]) / total_weight
        output.append(
            {
                "id": f"{session_id}-{index}",
                "start": round(cursor, 2),
                "end": round(max(cursor, end), 2),
                "text": part,
                "language": language or "zh",
                "speaker": 0,
                "source": "实时会议",
            }
        )
        cursor = max(cursor, end)
    return output


@app.on_event("startup")
def startup() -> None:
    global asr
    asr = Qwen3ASRModel.LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_new_tokens=32,
        enforce_eager=True,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    response_format: str = Form(default="json"),
    language: str | None = Form(default=None),
) -> dict[str, str]:
    if asr is None:
        raise HTTPException(status_code=503, detail="ASR model not ready")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    target = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        with target.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        forced_language = language.strip() if language else None
        with asr_lock:
            result = asr.transcribe(str(target), language=forced_language)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_error_detail(exc, "ASR 文件转写失败，模型没有返回具体错误。")) from exc
    finally:
        target.unlink(missing_ok=True)

    return {
        "text": getattr(result, "text", "") or "",
        "language": getattr(result, "language", "") or "",
    }


@app.post("/api/start")
async def start() -> dict[str, str]:
    if asr is None:
        raise HTTPException(status_code=503, detail="ASR model not ready")
    session_id = uuid.uuid4().hex
    try:
        state = await _run_asr_locked(
            asr.init_streaming_state,
            unfixed_chunk_num=UNFIXED_CHUNK_NUM,
            unfixed_token_num=UNFIXED_TOKEN_NUM,
            chunk_size_sec=CHUNK_SIZE_SEC,
            timeout=START_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="ASR 流式会话初始化超时。") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_error_detail(exc, "ASR 流式会话初始化失败。")) from exc
    now = time.time()
    with sessions_lock:
        SESSIONS[session_id] = Session(state=state, created_at=now, last_seen=now)
    return {"session_id": session_id}


@app.post("/api/chunk")
async def chunk(request: Request, session_id: str = Query(...)) -> dict[str, Any]:
    if asr is None:
        raise HTTPException(status_code=503, detail="ASR model not ready")
    session = _get_session(session_id)
    if request.headers.get("content-type", "").split(";")[0] != "application/octet-stream":
        raise HTTPException(status_code=400, detail="expect application/octet-stream")

    raw = await request.body()
    if len(raw) % 4 != 0:
        raise HTTPException(status_code=400, detail="float32 bytes length not multiple of 4")

    pcm = np.frombuffer(raw, dtype=np.float32).reshape(-1)
    session.audio_seconds += float(pcm.shape[0]) / SAMPLE_RATE
    try:
        await _run_asr_locked(asr.streaming_transcribe, pcm, session.state, timeout=CHUNK_TIMEOUT_SEC)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="ASR 流式识别超时。") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_error_detail(exc, "ASR 流式识别失败。")) from exc

    language = getattr(session.state, "language", "") or "zh"
    text = getattr(session.state, "text", "") or ""
    session.segments = _segments_from_text(session_id, text, session.audio_seconds, language)
    return {
        "language": language,
        "text": text,
        "segments": session.segments,
    }


@app.post("/api/finish")
async def finish(session_id: str = Query(...)) -> dict[str, Any]:
    if asr is None:
        raise HTTPException(status_code=503, detail="ASR model not ready")
    try:
        session = _get_session(session_id)
    except HTTPException as exc:
        if exc.status_code == 409:
            return {"language": "", "text": "", "segments": []}
        raise
    try:
        await _run_asr_locked(asr.finish_streaming_transcribe, session.state, timeout=FINISH_TIMEOUT_SEC)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="ASR 流式收尾超时。") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_error_detail(exc, "ASR 流式收尾失败。")) from exc
    language = getattr(session.state, "language", "") or "zh"
    text = getattr(session.state, "text", "") or ""
    session.segments = _segments_from_text(session_id, text, session.audio_seconds, language)
    output = {
        "language": language,
        "text": text,
        "segments": session.segments,
    }
    with sessions_lock:
        SESSIONS.pop(session_id, None)
    return output


@app.delete("/api/session")
def delete_session(session_id: str = Query(...)) -> dict[str, Any]:
    with sessions_lock:
        existed = SESSIONS.pop(session_id, None) is not None
    return {"released": existed, "session_id": session_id}


@app.post("/api/reset")
def reset_sessions(max_age_sec: float = Query(default=0)) -> dict[str, Any]:
    now = time.time()
    with sessions_lock:
        if max_age_sec <= 0:
            count = len(SESSIONS)
            SESSIONS.clear()
        else:
            dead = [sid for sid, item in SESSIONS.items() if now - item.last_seen >= max_age_sec]
            for sid in dead:
                SESSIONS.pop(sid, None)
            count = len(dead)
    return {"released": count}
