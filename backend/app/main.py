from __future__ import annotations

import shutil
import time
import uuid
import wave
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from .adapters import stream_chunk, stream_finish, stream_start, stream_status, summarize_meeting_cards, summarize_stage, summary_cards_to_text, transcribe_audio, write_minutes_document
from .schemas import AppSettings, AudioTranscriptionResult, Meeting, MeetingCreate, MeetingDetail, MinutesDocumentResult, StageSummary, StageSummaryResult, SummaryCards, SummaryResult, TranscriptSegment
from .store import (
    create_meeting,
    delete_meeting,
    get_meeting,
    ensure_meeting_storage,
    get_settings,
    read_db,
    register_meeting_file,
    replace_transcript,
    save_minutes_document,
    save_settings,
    save_summary,
    save_summary_cards,
    save_stage_summaries,
    update_meeting_status,
)


app = FastAPI(title="AI Meeting Transcription", version="1.0.0")

STREAM_SAMPLE_RATE = 16000
STREAM_CONTEXT: dict[str, dict[str, Any]] = {}
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


@app.get("/api/settings", response_model=AppSettings)
def get_app_settings() -> dict:
    return get_settings()


@app.put("/api/settings", response_model=AppSettings)
def put_app_settings(payload: AppSettings) -> dict:
    return save_settings(payload)


@app.get("/api/meetings", response_model=list[Meeting])
def list_meetings() -> list[dict]:
    return read_db()["meetings"]


@app.post("/api/meetings", response_model=Meeting)
def post_meeting(payload: MeetingCreate) -> dict:
    return create_meeting(payload)


@app.delete("/api/meetings/{meeting_id}")
def remove_meeting(meeting_id: str) -> dict[str, str]:
    if not delete_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    return {"status": "deleted"}


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
        "summary_cards": data["summary_cards"].get(meeting_id, {}),
        "stage_summaries": data["stage_summaries"].get(meeting_id, []),
        "minutes_document": data["minutes_documents"].get(meeting_id, ""),
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
        if not item.text or item.kind == "divider":
            continue
        speaker = f"说话人 {item.speaker}" if item.speaker is not None else "说话人 ?"
        source = f" [{item.source}]" if item.source else ""
        lines.append(f"[{_format_clock(item.start)}-{_format_clock(item.end)}] {speaker}{source}: {item.text}")
    return "\n".join(lines).strip()


def _window_text(segments: list[TranscriptSegment], start: float, end: float) -> str:
    return "\n".join(
        item.text
        for item in segments
        if item.text and item.kind != "divider" and start <= ((item.start + item.end) / 2) < end
    ).strip()


def _stage_summary_reference(items: list[StageSummary]) -> str:
    lines: list[str] = []
    for item in items:
        lines.append(f"{item.start:.0f}-{item.end:.0f}秒：{item.title}")
        for summary in item.summary:
            lines.append(f"- {summary}")
    return "\n".join(lines).strip()


def _dirty_stage_summary(item: StageSummary) -> bool:
    text = "\n".join([item.title, *item.summary, *item.conclusions, *item.todos]).lower()
    return "thinking process" in text or "analyze user input" in text or "mental refinement" in text or "思考过程" in text


def _clear_generated_content(meeting_id: str) -> None:
    save_summary(meeting_id, "")
    save_summary_cards(meeting_id, SummaryCards())
    save_stage_summaries(meeting_id, [])
    save_minutes_document(meeting_id, "")


def _error_detail(exc: Exception, fallback: str) -> str:
    return str(exc).strip() or fallback


def _speech_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    return [item for item in segments if item.kind != "divider" and item.text]


def _stream_divider(session_id: str, offset: float) -> TranscriptSegment:
    return TranscriptSegment(
        id=f"divider-{session_id}",
        start=round(offset, 2),
        end=round(offset, 2),
        text="新一段录音开始",
        language="zh",
        source="实时会议",
        kind="divider",
    )


def _offset_stream_segments(segments: list[TranscriptSegment], session_id: str, offset: float) -> list[TranscriptSegment]:
    output: list[TranscriptSegment] = []
    for index, segment in enumerate(segments):
        if segment.kind == "divider":
            continue
        output.append(
            segment.model_copy(
                update={
                    "id": f"{session_id}-{segment.id or index}",
                    "start": round(segment.start + offset, 2),
                    "end": round(segment.end + offset, 2),
                    "source": segment.source or "实时会议",
                    "kind": "speech",
                }
            )
        )
    return output


def _combined_stream_result(payload: dict, session_id: str) -> AudioTranscriptionResult:
    context = STREAM_CONTEXT.get(session_id, {})
    prefix = list(context.get("prefix") or [])
    offset = float(context.get("offset") or 0)
    current_result = _stream_result(payload, session_id)
    current_segments = _offset_stream_segments(current_result.segments, session_id, offset)
    segments = [*prefix, *current_segments]
    return AudioTranscriptionResult(
        text=_plain_text(segments),
        segments=segments,
        files=current_result.files,
        speaker_ready=current_result.speaker_ready,
        speaker_status=current_result.speaker_status,
    )


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
    _clear_generated_content(meeting_id)
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
    _clear_generated_content(meeting_id)
    update_meeting_status(meeting_id, f"批量转写完成（{len(file_names)} 个文件）")
    return AudioTranscriptionResult(text=_plain_text(segments), segments=segments, files=file_names)


@app.post("/api/meetings/{meeting_id}/summary", response_model=SummaryResult)
async def generate_summary(meeting_id: str) -> SummaryResult:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    data = read_db()
    transcript = data["transcripts"].get(meeting_id, [])
    text = _plain_text([TranscriptSegment(**item) for item in transcript])
    stage_summaries = [
        StageSummary(**item)
        for item in data["stage_summaries"].get(meeting_id, [])
        if item.get("summary") or item.get("conclusions") or item.get("todos")
    ]
    minutes_document = str(data["minutes_documents"].get(meeting_id, "") or "").strip()
    try:
        if not minutes_document:
            minutes_document = await write_minutes_document(meeting["title"], text, _stage_summary_reference(stage_summaries))
            save_minutes_document(meeting_id, minutes_document)
        summary_cards = await summarize_meeting_cards(meeting["title"], text, minutes_document)
        summary = summary_cards_to_text(summary_cards)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "智能纪要生成失败，后端没有返回具体错误。")) from exc
    save_summary(meeting_id, summary)
    save_summary_cards(meeting_id, summary_cards)
    update_meeting_status(meeting_id, "纪要已生成")
    return SummaryResult(summary=summary, summary_cards=summary_cards, minutes_document=minutes_document)


@app.post("/api/meetings/{meeting_id}/stage-summaries", response_model=StageSummaryResult)
async def generate_stage_summaries(
    meeting_id: str,
    window_seconds: int = Query(120, ge=30, le=600),
    refresh: bool = Query(False),
) -> StageSummaryResult:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    data = read_db()
    segments = [
        TranscriptSegment(**item)
        for item in data["transcripts"].get(meeting_id, [])
        if item.get("text") and item.get("kind") != "divider"
    ]
    if not segments:
        raise HTTPException(status_code=400, detail="当前会议还没有转写文字，无法生成阶段摘要。")

    existing = [
        StageSummary(**item)
        for item in data["stage_summaries"].get(meeting_id, [])
        if item.get("summary") or item.get("conclusions") or item.get("todos")
    ]
    existing = [item for item in existing if not _dirty_stage_summary(item)]
    if refresh:
        existing = []
    existing_windows = {(round(item.start, 1), round(item.end, 1)) for item in existing}
    max_end = max(item.end for item in segments)
    cursor = 0.0

    try:
        while cursor < max_end:
            end = min(cursor + float(window_seconds), max_end)
            text = _window_text(segments, cursor, end)
            window_key = (round(cursor, 1), round(end, 1))
            if text and window_key not in existing_windows:
                result = await summarize_stage(meeting["title"], cursor, end, text)
                existing.append(
                    StageSummary(
                        id=uuid.uuid4().hex,
                        start=round(cursor, 2),
                        end=round(end, 2),
                        title=result["title"],
                        summary=result["summary"],
                        conclusions=result["conclusions"],
                        todos=result["todos"],
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                )
                existing_windows.add(window_key)
            cursor += float(window_seconds)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "阶段摘要生成失败，后端没有返回具体错误。")) from exc

    existing.sort(key=lambda item: item.start)
    save_stage_summaries(meeting_id, existing)
    return StageSummaryResult(stage_summaries=existing)


@app.post("/api/meetings/{meeting_id}/minutes-document", response_model=MinutesDocumentResult)
async def generate_minutes_document(meeting_id: str) -> MinutesDocumentResult:
    meeting = get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    data = read_db()
    segments = [
        TranscriptSegment(**item)
        for item in data["transcripts"].get(meeting_id, [])
        if item.get("text") and item.get("kind") != "divider"
    ]
    text = _plain_text(segments)
    stage_summaries = [
        StageSummary(**item)
        for item in data["stage_summaries"].get(meeting_id, [])
        if item.get("summary") or item.get("conclusions") or item.get("todos")
    ]
    try:
        document = await write_minutes_document(meeting["title"], text, _stage_summary_reference(stage_summaries))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "正式纪要生成失败，后端没有返回具体错误。")) from exc

    save_minutes_document(meeting_id, document)
    update_meeting_status(meeting_id, "正式纪要已生成")
    return MinutesDocumentResult(document=document)


@app.post("/api/meetings/{meeting_id}/stream/start")
async def start_low_latency_stream(meeting_id: str) -> dict[str, str]:
    if not get_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        session_id = await stream_start()
        _start_stream_recording(meeting_id, session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_error_detail(exc, "流式 ASR 启动失败，后端没有返回具体错误。")) from exc

    existing = [
        TranscriptSegment(**item)
        for item in read_db()["transcripts"].get(meeting_id, [])
        if item.get("text")
    ]
    offset = max((item.end for item in _speech_segments(existing)), default=0.0)
    divider = _stream_divider(session_id, offset)
    prefix = [*existing, divider]
    STREAM_CONTEXT[session_id] = {
        "meeting_id": meeting_id,
        "offset": offset,
        "prefix": prefix,
    }
    replace_transcript(meeting_id, prefix)
    _clear_generated_content(meeting_id)
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

    result = _combined_stream_result(payload, session_id)
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

    result = _combined_stream_result(payload, session_id)
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

    result = _combined_stream_result(payload, session_id)
    result.files = [recorded_file] if recorded_file else []
    replace_transcript(meeting_id, result.segments)
    STREAM_CONTEXT.pop(session_id, None)
    _clear_generated_content(meeting_id)
    update_meeting_status(meeting_id, "转写完成")
    return result
