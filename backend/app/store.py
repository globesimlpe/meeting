from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .schemas import MeetingCreate, TranscriptSegment


ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = ROOT / "storage"
DB_PATH = STORAGE_DIR / "db.json"
MEETINGS_DIR = STORAGE_DIR / "meetings"

_lock = Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_db() -> dict[str, Any]:
    return {
        "meetings": [],
        "transcripts": {},
        "summaries": {},
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
    return data


def _write_meeting_db(
    meeting: dict[str, Any],
    transcript: list[dict[str, Any]] | None = None,
    summary: str | None = None,
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
                },
            )
            data["transcripts"][meeting_id] = meeting_data["transcript"]
            data["summaries"][meeting_id] = meeting_data["summary"]
        return data


def write_db(data: dict[str, Any]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        clean = normalize_db(data)
        index = {**clean, "transcripts": {}, "summaries": {}}
        DB_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        for meeting in clean["meetings"]:
            meeting_id = meeting["id"]
            _write_meeting_db(
                meeting,
                transcript=clean["transcripts"].get(meeting_id, []),
                summary=clean["summaries"].get(meeting_id, ""),
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
    write_db(data)
    ensure_meeting_storage(meeting["id"])
    return meeting


def get_meeting(meeting_id: str) -> dict[str, Any] | None:
    data = read_db()
    return next((item for item in data["meetings"] if item["id"] == meeting_id), None)


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
            files=files,
        )
