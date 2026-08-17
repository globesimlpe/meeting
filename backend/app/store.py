from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .schemas import AppSettings, MeetingCreate, StageSummary, SummaryCards, TranscriptSegment


ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = ROOT / "storage"
DB_PATH = STORAGE_DIR / "db.json"
MEETINGS_DIR = STORAGE_DIR / "meetings"

_lock = Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_settings() -> dict[str, Any]:
    return AppSettings(
        asr_base_url=os.getenv("ASR_BASE_URL", ""),
        asr_model=os.getenv("ASR_MODEL", "/home/regchen/Chuyi/models/Qwen3-ASR-0.6B"),
        asr_api_key=os.getenv("ASR_API_KEY", "EMPTY"),
        asr_transcribe_path=os.getenv("ASR_TRANSCRIBE_PATH", "/v1/audio/transcriptions"),
        asr_max_retries=int(os.getenv("ASR_MAX_RETRIES", "2")),
        asr_stream_base_url=os.getenv("ASR_STREAM_BASE_URL", "http://127.0.0.1:8005"),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_model=os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-NVFP4"),
        llm_api_key=os.getenv("LLM_API_KEY", "EMPTY"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1200")),
    ).model_dump()


def empty_db() -> dict[str, Any]:
    return {
        "meetings": [],
        "transcripts": {},
        "summaries": {},
        "summary_cards": {},
        "stage_summaries": {},
        "minutes_documents": {},
        "settings": default_settings(),
    }


def empty_meeting_db(meeting: dict[str, Any]) -> dict[str, Any]:
    return {
        "meeting": {
            "id": str(meeting["id"]),
            "title": str(meeting.get("title") or "未命名会议"),
            "status": str(meeting.get("status") or "已创建"),
            "created_at": str(meeting.get("created_at") or now_iso()),
        },
        "transcript": [],
        "summary": "",
        "summary_cards": SummaryCards().model_dump(),
        "stage_summaries": [],
        "minutes_document": "",
        "files": [],
    }


def normalize_db(data: dict[str, Any]) -> dict[str, Any]:
    clean = empty_db()
    clean["meetings"] = [
        {
            "id": str(item["id"]),
            "title": str(item.get("title") or "未命名会议"),
            "status": str(item.get("status") or "已创建"),
            "created_at": str(item.get("created_at") or now_iso()),
        }
        for item in data.get("meetings", [])
        if item.get("id")
    ]
    for meeting in clean["meetings"]:
        meeting_id = meeting["id"]
        clean["transcripts"][meeting_id] = [
            TranscriptSegment(**segment).model_dump()
            for segment in data.get("transcripts", {}).get(meeting_id, [])
            if isinstance(segment, dict) and segment.get("text")
        ]
        clean["summaries"][meeting_id] = str(data.get("summaries", {}).get(meeting_id, ""))
        clean["summary_cards"][meeting_id] = SummaryCards(**(data.get("summary_cards", {}).get(meeting_id, {}) or {})).model_dump()
        clean["stage_summaries"][meeting_id] = [
            StageSummary(**item).model_dump()
            for item in data.get("stage_summaries", {}).get(meeting_id, [])
            if isinstance(item, dict) and (item.get("summary") or item.get("conclusions") or item.get("todos"))
        ]
        clean["minutes_documents"][meeting_id] = str(data.get("minutes_documents", {}).get(meeting_id, ""))

    settings = default_settings()
    settings.update(data.get("settings", {}) or {})
    clean["settings"] = AppSettings(**settings).model_dump()
    return clean


def normalize_meeting_db(data: dict[str, Any], meeting: dict[str, Any]) -> dict[str, Any]:
    clean = empty_meeting_db(meeting)
    saved_meeting = data.get("meeting") if isinstance(data.get("meeting"), dict) else {}
    clean["meeting"] = {
        **clean["meeting"],
        "title": str(saved_meeting.get("title") or clean["meeting"]["title"]),
        "status": str(saved_meeting.get("status") or clean["meeting"]["status"]),
        "created_at": str(saved_meeting.get("created_at") or clean["meeting"]["created_at"]),
    }
    clean["transcript"] = [
        TranscriptSegment(**segment).model_dump()
        for segment in data.get("transcript", [])
        if isinstance(segment, dict) and segment.get("text")
    ]
    clean["summary"] = str(data.get("summary") or "")
    clean["summary_cards"] = SummaryCards(**(data.get("summary_cards") or {})).model_dump()
    clean["stage_summaries"] = [
        StageSummary(**item).model_dump()
        for item in data.get("stage_summaries", [])
        if isinstance(item, dict) and (item.get("summary") or item.get("conclusions") or item.get("todos"))
    ]
    clean["minutes_document"] = str(data.get("minutes_document") or "")
    clean["files"] = [str(item) for item in data.get("files", []) if item]
    return clean


def meeting_dir(meeting_id: str) -> Path:
    return MEETINGS_DIR / meeting_id


def meeting_db_path(meeting_id: str) -> Path:
    return meeting_dir(meeting_id) / "db.json"


def ensure_meeting_storage(meeting_id: str) -> Path:
    target = meeting_dir(meeting_id)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _read_meeting_db(meeting: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    path = meeting_db_path(str(meeting["id"]))
    if path.exists():
        return normalize_meeting_db(json.loads(path.read_text(encoding="utf-8")), meeting)

    data = empty_meeting_db(meeting)
    if fallback:
        data["transcript"] = [
            TranscriptSegment(**segment).model_dump()
            for segment in fallback.get("transcript", [])
            if isinstance(segment, dict) and segment.get("text")
        ]
        data["summary"] = str(fallback.get("summary") or "")
        data["summary_cards"] = SummaryCards(**(fallback.get("summary_cards") or {})).model_dump()
        data["stage_summaries"] = [
            StageSummary(**item).model_dump()
            for item in fallback.get("stage_summaries", [])
            if isinstance(item, dict) and (item.get("summary") or item.get("conclusions") or item.get("todos"))
        ]
        data["minutes_document"] = str(fallback.get("minutes_document") or "")
    return data


def _write_meeting_db(
    meeting: dict[str, Any],
    transcript: list[dict[str, Any]] | None = None,
    summary: str | None = None,
    summary_cards: dict[str, Any] | None = None,
    stage_summaries: list[dict[str, Any]] | None = None,
    minutes_document: str | None = None,
    files: list[str] | None = None,
) -> None:
    meeting_id = str(meeting["id"])
    path = meeting_db_path(meeting_id)
    current = _read_meeting_db(meeting) if path.exists() else empty_meeting_db(meeting)
    current["meeting"] = {
        "id": str(meeting["id"]),
        "title": str(meeting.get("title") or current["meeting"]["title"]),
        "status": str(meeting.get("status") or current["meeting"]["status"]),
        "created_at": str(meeting.get("created_at") or current["meeting"]["created_at"]),
    }
    if transcript is not None:
        current["transcript"] = [
            TranscriptSegment(**segment).model_dump()
            for segment in transcript
            if isinstance(segment, dict) and segment.get("text")
        ]
    if summary is not None:
        current["summary"] = str(summary)
    if summary_cards is not None:
        current["summary_cards"] = SummaryCards(**summary_cards).model_dump()
    if stage_summaries is not None:
        current["stage_summaries"] = [
            StageSummary(**item).model_dump()
            for item in stage_summaries
            if isinstance(item, dict) and (item.get("summary") or item.get("conclusions") or item.get("todos"))
        ]
    if minutes_document is not None:
        current["minutes_document"] = str(minutes_document)
    if files is not None:
        current["files"] = [str(item) for item in files if item]

    ensure_meeting_storage(meeting_id)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        DB_PATH.write_text(json.dumps(empty_db(), ensure_ascii=False, indent=2), encoding="utf-8")


def read_db() -> dict[str, Any]:
    ensure_storage()
    with _lock:
        legacy = normalize_db(json.loads(DB_PATH.read_text(encoding="utf-8")))
        data = empty_db()
        data["meetings"] = legacy["meetings"]
        for meeting in data["meetings"]:
            meeting_id = meeting["id"]
            meeting_data = _read_meeting_db(
                meeting,
                {
                    "transcript": legacy["transcripts"].get(meeting_id, []),
                    "summary": legacy["summaries"].get(meeting_id, ""),
                    "summary_cards": legacy["summary_cards"].get(meeting_id, {}),
                    "stage_summaries": legacy["stage_summaries"].get(meeting_id, []),
                    "minutes_document": legacy["minutes_documents"].get(meeting_id, ""),
                },
            )
            data["transcripts"][meeting_id] = meeting_data["transcript"]
            data["summaries"][meeting_id] = meeting_data["summary"]
            data["summary_cards"][meeting_id] = meeting_data["summary_cards"]
            data["stage_summaries"][meeting_id] = meeting_data["stage_summaries"]
            data["minutes_documents"][meeting_id] = meeting_data["minutes_document"]
        data["settings"] = legacy["settings"]
        return data


def write_db(data: dict[str, Any]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        clean = normalize_db(data)
        index = {
            **clean,
            "transcripts": {},
            "summaries": {},
            "summary_cards": {},
            "stage_summaries": {},
            "minutes_documents": {},
        }
        DB_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        for meeting in clean["meetings"]:
            meeting_id = meeting["id"]
            _write_meeting_db(
                meeting,
                transcript=clean["transcripts"].get(meeting_id, []),
                summary=clean["summaries"].get(meeting_id, ""),
                summary_cards=clean["summary_cards"].get(meeting_id, {}),
                stage_summaries=clean["stage_summaries"].get(meeting_id, []),
                minutes_document=clean["minutes_documents"].get(meeting_id, ""),
            )


def create_meeting(payload: MeetingCreate) -> dict[str, Any]:
    data = read_db()
    meeting = {
        "id": uuid.uuid4().hex,
        "title": payload.title.strip(),
        "status": "已创建",
        "created_at": now_iso(),
    }
    data["meetings"].insert(0, meeting)
    data["transcripts"][meeting["id"]] = []
    data["summaries"][meeting["id"]] = ""
    data["summary_cards"][meeting["id"]] = SummaryCards().model_dump()
    data["stage_summaries"][meeting["id"]] = []
    data["minutes_documents"][meeting["id"]] = ""
    write_db(data)
    ensure_meeting_storage(meeting["id"])
    return meeting


def get_meeting(meeting_id: str) -> dict[str, Any] | None:
    data = read_db()
    return next((item for item in data["meetings"] if item["id"] == meeting_id), None)


def delete_meeting(meeting_id: str) -> bool:
    data = read_db()
    before_count = len(data["meetings"])
    data["meetings"] = [item for item in data["meetings"] if item["id"] != meeting_id]
    if len(data["meetings"]) == before_count:
        return False

    data["transcripts"].pop(meeting_id, None)
    data["summaries"].pop(meeting_id, None)
    data["summary_cards"].pop(meeting_id, None)
    data["stage_summaries"].pop(meeting_id, None)
    data["minutes_documents"].pop(meeting_id, None)
    write_db(data)

    shutil.rmtree(meeting_dir(meeting_id), ignore_errors=True)
    return True


def update_meeting_status(meeting_id: str, status: str) -> None:
    data = read_db()
    for meeting in data["meetings"]:
        if meeting["id"] == meeting_id:
            meeting["status"] = status
            break
    write_db(data)


def replace_transcript(meeting_id: str, segments: list[TranscriptSegment]) -> None:
    data = read_db()
    data["transcripts"][meeting_id] = [item.model_dump() for item in segments]
    write_db(data)


def save_summary(meeting_id: str, summary: str) -> None:
    data = read_db()
    data["summaries"][meeting_id] = summary
    write_db(data)


def save_summary_cards(meeting_id: str, summary_cards: SummaryCards) -> None:
    data = read_db()
    data["summary_cards"][meeting_id] = summary_cards.model_dump()
    write_db(data)


def save_stage_summaries(meeting_id: str, stage_summaries: list[StageSummary]) -> None:
    data = read_db()
    data["stage_summaries"][meeting_id] = [item.model_dump() for item in stage_summaries]
    write_db(data)


def save_minutes_document(meeting_id: str, document: str) -> None:
    data = read_db()
    data["minutes_documents"][meeting_id] = document
    write_db(data)


def get_settings() -> dict[str, Any]:
    return read_db()["settings"]


def save_settings(settings: AppSettings) -> dict[str, Any]:
    data = read_db()
    data["settings"] = settings.model_dump()
    write_db(data)
    return data["settings"]


def register_meeting_file(meeting_id: str, file_name: str) -> None:
    meeting = get_meeting(meeting_id)
    if not meeting:
        return
    with _lock:
        meeting_data = _read_meeting_db(meeting)
        files = list(meeting_data.get("files") or [])
        if file_name not in files:
            files.append(file_name)
        _write_meeting_db(
            meeting,
            transcript=meeting_data["transcript"],
            summary=meeting_data["summary"],
            summary_cards=meeting_data["summary_cards"],
            stage_summaries=meeting_data["stage_summaries"],
            minutes_document=meeting_data["minutes_document"],
            files=files,
        )
