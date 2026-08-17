from __future__ import annotations

import shutil
import uuid
import wave
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from .adapters import stream_chunk, stream_finish, stream_start, stream_status, summarize_meeting, transcribe_audio
from .schemas import AudioTranscriptionResult, Meeting, MeetingCreate, MeetingDetail, SummaryResult, TranscriptSegment
from .store import (
    create_meeting,
    get_meeting,
    ensure_meeting_storage,
    read_db,
    register_meeting_file,
    replace_transcript,
    save_summary,
    update_meeting_status,
)


app = FastAPI(title="AI Meeting Transcription", version="1.0.0")

STREAM_SAMPLE_RATE = 16000
_stream_recordings: dict[str, dict[str, Any]] = {}
_stream_recordings_lock = Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meetings", response_model=list[Meeting])
def list_meetings() -> list[dict]:
    return read_db()["meetings"]


@app.post("/api/meetings", response_model=Meeting)
def post_meeting(payload: MeetingCreate) -> dict:
    return create_meeting(payload)


@app.get("/api/meetings/{meeting_id}", response_model=MeetingDetail)
def get_meeting_detail(meeting_id: str) -> dict:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    data = read_db()
    return {
        "meeting": meeting,
        "transcript": data["transcripts"].get(meeting_id, []),
        "summary": data["summaries"].get(meeting_id, ""),
    }


def _save_upload(meeting_id: str, file: UploadFile, prefix: str) -> Path:
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    target_dir = ensure_meeting_storage(meeting_id)
    target = target_dir / f"{prefix}-{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    register_meeting_file(meeting_id, target.name)
    return target


def _stream_recording_paths(meeting_id: str, session_id: str) -> tuple[Path, Path]:
    target_dir = ensure_meeting_storage(meeting_id)
    safe_session_id = "".join(char for char in session_id if char.isalnum()) or uuid.uuid4().hex
    raw_path = target_dir / f".stream-{safe_session_id}.f32.tmp"
    wav_path = target_dir / f"stream-{safe_session_id}.wav"
    return raw_path, wav_path


def _start_stream_recording(meeting_id: str, session_id: str) -> None:
    raw_path, wav_path = _stream_recording_paths(meeting_id, session_id)
    raw_path.write_bytes(b"")
    with _stream_recordings_lock:
        _stream_recordings[session_id] = {
            "meeting_id": meeting_id,
            "raw_path": raw_path,
            "wav_path": wav_path,
        }


def _append_stream_recording(meeting_id: str, session_id: str, pcm: bytes) -> None:
    if len(pcm) % 4 != 0:
        raise HTTPException(status_code=400, detail="实时音频数据格式错误：Float32 PCM 字节长度不是 4 的倍数。")
    with _stream_recordings_lock:
        recording = _stream_recordings.get(session_id)
        if not recording or recording.get("meeting_id") != meeting_id:
            return
        raw_path = Path(recording["raw_path"])
        with raw_path.open("ab") as output:
            output.write(pcm)


def _write_stream_wav(raw_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("rb") as source, wave.open(str(wav_path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(STREAM_SAMPLE_RATE)
        while True:
            chunk = source.read(STREAM_SAMPLE_RATE * 4 * 30)
            if not chunk:
                break
            usable_length = len(chunk) - (len(chunk) % 4)
            if usable_length <= 0:
                continue
            samples = np.frombuffer(chunk[:usable_length], dtype="<f4")
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
            samples = np.clip(samples, -1.0, 1.0)
            target.writeframes((samples * 32767).astype("<i2").tobytes())


def _finish_stream_recording(meeting_id: str, session_id: str) -> str | None:
    with _stream_recordings_lock:
        recording = _stream_recordings.pop(session_id, None)
    if not recording or recording.get("meeting_id") != meeting_id:
        return None

    raw_path = Path(recording["raw_path"])
    wav_path = Path(recording["wav_path"])
    try:
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            return None
        _write_stream_wav(raw_path, wav_path)
        register_meeting_file(meeting_id, wav_path.name)
        return wav_path.name
    finally:
        raw_path.unlink(missing_ok=True)


def _format_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}" if hours else f"{minutes:02d}:{sec:02d}"


def _plain_text(segments: list[TranscriptSegment]) -> str:
    lines = []
    for item in segments:
        if not item.text:
            continue
        speaker = f"说话人 {item.speaker}" if item.speaker is not None else "说话人 ?"
        source = f" [{item.source}]" if item.source else ""
        lines.append(f"[{_format_clock(item.start)}-{_format_clock(item.end)}] {speaker}{source}: {item.text}")
    return "\n".join(lines).strip()


def _error_detail(exc: Exception, fallback: str) -> str:
    return str(exc).strip() or fallback


def _segments_from_stream_payload(payload: dict, session_id: str) -> list[TranscriptSegment]:
    raw_segments = payload.get("segments")
    if isinstance(raw_segments, list) and raw_segments:
        segments: list[TranscriptSegment] = []
        for index, item in enumerate(raw_segments):
            try:
                segment = TranscriptSegment(**item)
            except Exception:
                continue
            if not segment.text.strip():
                continue
            if not segment.id:
                segment.id = f"{session_id}-{index}"
            segments.append(segment)
        return segments

    text = str(payload.get("text") or "").strip()
    if not text:
        return []
    return [
        TranscriptSegment(
            id=session_id,
            start=float(payload.get("start") or 0),
            end=float(payload.get("end") or 0),
            text=text,
            language=payload.get("language") or "zh",
            speaker=0,
            source="实时会议",
        )
    ]


def _stream_result(payload: dict, session_id: str) -> AudioTranscriptionResult:
    segments = _segments_from_stream_payload(payload, session_id)
    text = _plain_text(segments)
    return AudioTranscriptionResult(
        text=text,
        segments=segments,
        speaker_ready=bool(payload.get("speaker_ready")),
        speaker_status=str(payload.get("speaker_status") or ""),
    )


@app.post("/api/meetings/{meeting_id}/audio", response_model=AudioTranscriptionResult)
async def upload_audio(meeting_id: str, file: UploadFile = File(...)) -> AudioTranscriptionResult:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")

    target = _save_upload(meeting_id, file, "upload")
    update_meeting_status(meeting_id, "转写中")
    try:
        segments = await transcribe_audio(target, source_name=file.filename or target.name)
    except Exception as exc:
        update_meeting_status(meeting_id, "转写失败")
        raise HTTPException(status_code=502, detail=_error_detail(exc, "音频转写失败，后端没有返回具体错误。")) from exc

    replace_transcript(meeting_id, segments)
    update_meeting_status(meeting_id, "转写完成")
    return AudioTranscriptionResult(text=_plain_text(segments), segments=segments, files=[file.filename or target.name])


@app.post("/api/meetings/{meeting_id}/audio/batch", response_model=AudioTranscriptionResult)
async def upload_audio_batch(meeting_id: str, files: list[UploadFile] = File(...)) -> AudioTranscriptionResult:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    if not files:
        raise HTTPException(status_code=400, detail="请至少选择一个音频文件。")

    update_meeting_status(meeting_id, "批量转写中")
    segments: list[TranscriptSegment] = []
    file_names: list[str] = []
    try:
        for index, upload in enumerate(files, start=1):
            file_name = upload.filename or f"audio-{index}.wav"
            target = _save_upload(meeting_id, upload, f"batch-{index}")
            file_names.append(file_name)
            segments.extend(await transcribe_audio(target, source_name=file_name))
    except Exception as exc:
        update_meeting_status(meeting_id, "批量转写失败")
        raise HTTPException(status_code=502, detail=_error_detail(exc, "批量音频转写失败，后端没有返回具体错误。")) from exc

    segments.sort(key=lambda item: (file_names.index(item.source) if item.source in file_names else len(file_names), item.start, item.end))
    replace_transcript(meeting_id, segments)
    update_meeting_status(meeting_id, f"批量转写完成（{len(file_names)} 个文件）")
    return AudioTranscriptionResult(text=_plain_text(segments), segments=segments, files=file_names)


@app.post("/api/meetings/{meeting_id}/summary", response_model=SummaryResult)
async def generate_summary(meeting_id: str) -> SummaryResult:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    transcript = read_db()["transcripts"].get(meeting_id, [])
    text = _plain_text([TranscriptSegment(**item) for item in transcript])
    try:
        summary = await summarize_meeting(meeting["title"], text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "AI 纪要生成失败，后端没有返回具体错误。")) from exc
    save_summary(meeting_id, summary)
    update_meeting_status(meeting_id, "纪要已生成")
    return SummaryResult(summary=summary)


@app.post("/api/meetings/{meeting_id}/stream/start")
async def start_low_latency_stream(meeting_id: str) -> dict[str, str]:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        session_id = await stream_start()
        _start_stream_recording(meeting_id, session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "流式 ASR 启动失败，后端没有返回具体错误。")) from exc
    replace_transcript(meeting_id, [])
    update_meeting_status(meeting_id, "实时转写中")
    return {"session_id": session_id}


@app.post("/api/meetings/{meeting_id}/stream/chunk", response_model=AudioTranscriptionResult)
async def push_low_latency_stream_chunk(
    meeting_id: str,
    request: Request,
    session_id: str = Query(...),
) -> AudioTranscriptionResult:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        body = await request.body()
        _append_stream_recording(meeting_id, session_id, body)
        payload = await stream_chunk(session_id, body)
    except HTTPException:
        raise
    except Exception as exc:
        update_meeting_status(meeting_id, "实时转写失败")
        raise HTTPException(status_code=502, detail=_error_detail(exc, "流式 ASR 识别失败，后端没有返回具体错误。")) from exc

    result = _stream_result(payload, session_id)
    if result.segments:
        replace_transcript(meeting_id, result.segments)
    return result


@app.get("/api/meetings/{meeting_id}/stream/status", response_model=AudioTranscriptionResult)
async def get_low_latency_stream_status(meeting_id: str, session_id: str = Query(...)) -> AudioTranscriptionResult:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        payload = await stream_status(session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "流式 ASR 状态读取失败，后端没有返回具体错误。")) from exc

    result = _stream_result(payload, session_id)
    if result.segments:
        replace_transcript(meeting_id, result.segments)
    return result


@app.post("/api/meetings/{meeting_id}/stream/finish", response_model=AudioTranscriptionResult)
async def finish_low_latency_stream(meeting_id: str, session_id: str = Query(...)) -> AudioTranscriptionResult:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    recorded_file = _finish_stream_recording(meeting_id, session_id)
    try:
        payload = await stream_finish(session_id)
    except Exception as exc:
        update_meeting_status(meeting_id, "实时转写失败")
        raise HTTPException(status_code=502, detail=_error_detail(exc, "流式 ASR 收尾失败，后端没有返回具体错误。")) from exc

    result = _stream_result(payload, session_id)
    result.files = [recorded_file] if recorded_file else []
    replace_transcript(meeting_id, result.segments)
    update_meeting_status(meeting_id, "转写完成")
    return result
