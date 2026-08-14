from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .adapters import stream_chunk, stream_finish, stream_start, summarize_meeting, transcribe_audio
from .schemas import AudioTranscriptionResult, Meeting, MeetingCreate, MeetingDetail, SummaryResult, TranscriptSegment
from .store import (
    UPLOAD_DIR,
    create_meeting,
    get_meeting,
    read_db,
    replace_transcript,
    save_summary,
    update_meeting_status,
)


app = FastAPI(title="AI Meeting Transcription", version="1.0.0")

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
    target = UPLOAD_DIR / f"{meeting_id}-{prefix}-{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return target


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
        payload = await stream_chunk(session_id, await request.body())
    except Exception as exc:
        update_meeting_status(meeting_id, "实时转写失败")
        raise HTTPException(status_code=502, detail=_error_detail(exc, "流式 ASR 识别失败，后端没有返回具体错误。")) from exc

    segments = _segments_from_stream_payload(payload, session_id)
    text = _plain_text(segments)
    replace_transcript(meeting_id, segments)
    return AudioTranscriptionResult(text=text, segments=segments)


@app.post("/api/meetings/{meeting_id}/stream/finish", response_model=AudioTranscriptionResult)
async def finish_low_latency_stream(meeting_id: str, session_id: str = Query(...)) -> AudioTranscriptionResult:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        payload = await stream_finish(session_id)
    except Exception as exc:
        update_meeting_status(meeting_id, "实时转写失败")
        raise HTTPException(status_code=502, detail=_error_detail(exc, "流式 ASR 收尾失败，后端没有返回具体错误。")) from exc

    segments = _segments_from_stream_payload(payload, session_id)
    text = _plain_text(segments)
    replace_transcript(meeting_id, segments)
    update_meeting_status(meeting_id, "转写完成")
    return AudioTranscriptionResult(text=text, segments=segments)
