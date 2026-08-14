from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
import uuid
import wave
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv

from .schemas import TranscriptSegment


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ASR_BASE_URL = os.getenv("ASR_BASE_URL", "").rstrip("/")
ASR_API_KEY = os.getenv("ASR_API_KEY", "EMPTY")
ASR_MODEL = os.getenv("ASR_MODEL", "/home/regchen/Chuyi/models/Qwen3-ASR-0.6B")
ASR_TRANSCRIBE_PATH = os.getenv("ASR_TRANSCRIBE_PATH", "/v1/audio/transcriptions")
ASR_MAX_RETRIES = int(os.getenv("ASR_MAX_RETRIES", "2"))
ASR_STREAM_BASE_URL = os.getenv("ASR_STREAM_BASE_URL", "http://127.0.0.1:8005").rstrip("/")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-NVFP4")

OFFLINE_ASR_PROVIDER = os.getenv("OFFLINE_ASR_PROVIDER", "funasr").lower()
FUNASR_MODEL = os.getenv("FUNASR_MODEL", "paraformer-zh")
FUNASR_VAD_MODEL = os.getenv("FUNASR_VAD_MODEL", "fsmn-vad")
FUNASR_PUNC_MODEL = os.getenv("FUNASR_PUNC_MODEL", "ct-punc")
FUNASR_SPK_MODEL = os.getenv("FUNASR_SPK_MODEL", "cam++")
FUNASR_DEVICE = os.getenv("FUNASR_DEVICE", "cuda:0")
FUNASR_BATCH_SIZE_S = int(os.getenv("FUNASR_BATCH_SIZE_S", "300"))
FUNASR_MERGE_LENGTH_S = int(os.getenv("FUNASR_MERGE_LENGTH_S", "15"))
REALTIME_ASR_PROVIDER = os.getenv("REALTIME_ASR_PROVIDER", "funasr").lower()
REALTIME_FUNASR_INTERVAL_SEC = float(os.getenv("REALTIME_FUNASR_INTERVAL_SEC", "3.0"))
REALTIME_FUNASR_MIN_AUDIO_SEC = float(os.getenv("REALTIME_FUNASR_MIN_AUDIO_SEC", "1.5"))
REALTIME_FUNASR_TMP_DIR = Path(os.getenv("REALTIME_FUNASR_TMP_DIR", "/tmp/meeting-ai-funasr-stream"))

_funasr_model: Any | None = None
_funasr_lock = Lock()
_realtime_lock = Lock()
_realtime_sessions: dict[str, dict[str, Any]] = {}


def _require_asr() -> str:
    if not ASR_BASE_URL:
        raise RuntimeError("未配置 ASR_BASE_URL，无法进行 Qwen 文件转写。")
    return f"{ASR_BASE_URL}{ASR_TRANSCRIBE_PATH}"


def _speaker_from_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None


def _seconds(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if number > 1000:
        number /= 1000
    return round(number, 2)


def _segments_from_payload(payload: dict, offset: float = 0, source: str = "") -> list[TranscriptSegment]:
    language = str(payload.get("language") or payload.get("detected_language") or "zh")
    raw_segments = payload.get("segments")
    if isinstance(raw_segments, list) and raw_segments:
        segments = []
        for item in raw_segments:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    id=str(item.get("id") or uuid.uuid4().hex),
                    start=round(float(item.get("start", 0)) + offset, 2),
                    end=round(float(item.get("end", 0)) + offset, 2),
                    text=text,
                    language=str(item.get("language") or language),
                    speaker=_speaker_from_value(item.get("speaker", item.get("spk"))),
                    source=source,
                )
            )
        return segments

    text = _clean_asr_text(str(payload.get("text") or payload.get("transcript") or ""))
    if not text:
        return []
    return [
        TranscriptSegment(
            id=uuid.uuid4().hex,
            start=round(offset, 2),
            end=round(offset, 2),
            text=text,
            language=language,
            source=source,
        )
    ]


def _segments_from_funasr_item(item: dict[str, Any], source: str) -> list[TranscriptSegment]:
    language = str(item.get("language") or "zh")
    sentence_info = item.get("sentence_info")
    if isinstance(sentence_info, list) and sentence_info:
        segments: list[TranscriptSegment] = []
        for sentence in sentence_info:
            text = str(sentence.get("sentence") or sentence.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    id=str(sentence.get("id") or uuid.uuid4().hex),
                    start=_seconds(sentence.get("start")),
                    end=_seconds(sentence.get("end")),
                    text=text,
                    language=language,
                    speaker=_speaker_from_value(sentence.get("spk", sentence.get("speaker"))),
                    source=source,
                )
            )
        return segments

    text = _clean_asr_text(str(item.get("text") or ""))
    if not text:
        return []
    return [
        TranscriptSegment(
            id=uuid.uuid4().hex,
            start=0,
            end=0,
            text=text,
            language=language,
            speaker=_speaker_from_value(item.get("spk", item.get("speaker"))),
            source=source,
        )
    ]


def _segments_from_funasr_payload(payload: Any, source: str) -> list[TranscriptSegment]:
    items = payload if isinstance(payload, list) else [payload]
    segments: list[TranscriptSegment] = []
    for item in items:
        if isinstance(item, dict):
            segments.extend(_segments_from_funasr_item(item, source))
    segments.sort(key=lambda item: (item.start, item.end, item.id))
    return segments


def _clean_asr_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^language\s+\w+\s*", "", text, flags=re.I)
    text = re.sub(r"</?asr_text>", "", text, flags=re.I)
    return text.strip()


def _clean_llm_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S | re.I)
    return text.strip()


def _asr_error(response: httpx.Response) -> RuntimeError:
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message") or payload.get("detail") or response.text
    except Exception:
        message = response.text
    return RuntimeError(f"ASR 服务返回 {response.status_code}: {message}")


def _convert_to_wav(file_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("服务器未安装 ffmpeg，无法把浏览器录音转换为 ASR 支持的 wav。")

    target = file_path.with_name(f"{file_path.stem}-asr.wav")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(file_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"音频转换失败: {(result.stderr or result.stdout).strip()}")
    return target


def _get_funasr_model() -> Any:
    global _funasr_model
    with _funasr_lock:
        if _funasr_model is None:
            try:
                from funasr import AutoModel
            except Exception as exc:
                raise RuntimeError("meeting-ai 环境未安装 FunASR，请先安装 funasr/modelscope/torch。") from exc

            _funasr_model = AutoModel(
                model=FUNASR_MODEL,
                vad_model=FUNASR_VAD_MODEL,
                punc_model=FUNASR_PUNC_MODEL,
                spk_model=FUNASR_SPK_MODEL,
                device=FUNASR_DEVICE,
                disable_update=True,
            )
        return _funasr_model


def _transcribe_audio_funasr_sync(file_path: Path, source: str) -> list[TranscriptSegment]:
    asr_file = _convert_to_wav(file_path)
    model = _get_funasr_model()
    payload = model.generate(
        input=str(asr_file),
        batch_size_s=FUNASR_BATCH_SIZE_S,
        merge_vad=True,
        merge_length_s=FUNASR_MERGE_LENGTH_S,
        sentence_timestamp=True,
    )
    return _segments_from_funasr_payload(payload, source=source)


async def _transcribe_audio_qwen(file_path: Path, offset: float = 0) -> list[TranscriptSegment]:
    url = _require_asr()
    headers = {"Authorization": f"Bearer {ASR_API_KEY}"}
    last_error: Exception | None = None
    asr_file = _convert_to_wav(file_path)

    for _attempt in range(ASR_MAX_RETRIES + 1):
        try:
            with asr_file.open("rb") as audio:
                files = {"file": (asr_file.name, audio, "audio/wav")}
                data = {
                    "model": ASR_MODEL,
                    "response_format": "json",
                }
                async with httpx.AsyncClient(timeout=600) as client:
                    response = await client.post(url, headers=headers, files=files, data=data)
                    if response.status_code >= 400:
                        raise _asr_error(response)
                    payload = response.json()
            return _segments_from_payload(payload, offset=offset, source=file_path.name)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"ASR 转写失败，已重试 {ASR_MAX_RETRIES} 次：{last_error}") from last_error


async def transcribe_audio(file_path: Path, offset: float = 0, source_name: str | None = None) -> list[TranscriptSegment]:
    source = source_name or file_path.name
    if OFFLINE_ASR_PROVIDER == "qwen":
        segments = await _transcribe_audio_qwen(file_path, offset=offset)
        return [item.model_copy(update={"source": source}) for item in segments]
    return await asyncio.to_thread(_transcribe_audio_funasr_sync, file_path, source)


def _text_from_segments(segments: list[TranscriptSegment]) -> str:
    return "\n".join(item.text for item in segments if item.text).strip()


def _pcm_duration(pcm_bytes: bytes) -> float:
    return len(pcm_bytes) / 4 / 16000


def _write_float32_wav(pcm_bytes: bytes, target: Path) -> None:
    samples = np.frombuffer(pcm_bytes, dtype=np.float32).reshape(-1)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    samples = np.clip(samples, -1.0, 1.0)
    int16 = (samples * 32767).astype(np.int16)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(int16.tobytes())


def _transcribe_realtime_funasr_sync(session_id: str, pcm_bytes: bytes) -> list[TranscriptSegment]:
    target = REALTIME_FUNASR_TMP_DIR / f"{session_id}.wav"
    _write_float32_wav(pcm_bytes, target)
    model = _get_funasr_model()
    payload = model.generate(
        input=str(target),
        batch_size_s=FUNASR_BATCH_SIZE_S,
        merge_vad=True,
        merge_length_s=FUNASR_MERGE_LENGTH_S,
        sentence_timestamp=True,
    )
    return _segments_from_funasr_payload(payload, source="实时会议")


def _realtime_payload(segments: list[TranscriptSegment]) -> dict[str, Any]:
    return {
        "language": segments[0].language if segments else "zh",
        "text": _text_from_segments(segments),
        "segments": [item.model_dump() for item in segments],
    }


async def _stream_start_funasr() -> str:
    session_id = uuid.uuid4().hex
    now = time.time()
    with _realtime_lock:
        _realtime_sessions[session_id] = {
            "pcm": bytearray(),
            "created_at": now,
            "updated_at": now,
            "last_generated_duration": 0.0,
            "segments": [],
        }
    return session_id


async def _stream_chunk_funasr(session_id: str, pcm: bytes) -> dict[str, Any]:
    if len(pcm) % 4 != 0:
        raise RuntimeError("实时音频数据格式错误：float32 bytes length not multiple of 4。")
    with _realtime_lock:
        session = _realtime_sessions.get(session_id)
        if not session:
            return {"language": "", "text": "", "segments": []}
        session["pcm"].extend(pcm)
        session["updated_at"] = time.time()
        pcm_bytes = bytes(session["pcm"])
        duration = _pcm_duration(pcm_bytes)
        last_duration = float(session.get("last_generated_duration") or 0)
        cached_segments = list(session.get("segments") or [])

    if duration < REALTIME_FUNASR_MIN_AUDIO_SEC or duration - last_duration < REALTIME_FUNASR_INTERVAL_SEC:
        return _realtime_payload(cached_segments)

    segments = await asyncio.to_thread(_transcribe_realtime_funasr_sync, session_id, pcm_bytes)
    with _realtime_lock:
        session = _realtime_sessions.get(session_id)
        if session is not None:
            session["segments"] = segments
            session["last_generated_duration"] = duration
    return _realtime_payload(segments)


async def _stream_finish_funasr(session_id: str) -> dict[str, Any]:
    with _realtime_lock:
        session = _realtime_sessions.pop(session_id, None)
    if not session:
        return {"language": "", "text": "", "segments": []}
    pcm_bytes = bytes(session.get("pcm") or b"")
    cached_segments = list(session.get("segments") or [])
    if not pcm_bytes or _pcm_duration(pcm_bytes) < REALTIME_FUNASR_MIN_AUDIO_SEC:
        return _realtime_payload(cached_segments)
    segments = await asyncio.to_thread(_transcribe_realtime_funasr_sync, session_id, pcm_bytes)
    return _realtime_payload(segments)


async def summarize_meeting(title: str, transcript_text: str) -> str:
    if not LLM_BASE_URL:
        raise RuntimeError("未配置 LLM_BASE_URL，无法生成真实 AI 纪要。")
    if not transcript_text.strip():
        raise RuntimeError("当前会议还没有转写文字，无法生成 AI 纪要。")

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是会议纪要助手。基于多人会议逐句转写生成简洁中文会议纪要，只输出正文，不要 Markdown 代码块。",
            },
            {
                "role": "user",
                "content": (
                    f"会议标题：{title}\n\n"
                    "请输出：1. 会议摘要；2. 关键结论；3. 待跟进事项。"
                    "没有明确内容的部分写“暂无”。可利用说话人和时间戳判断上下文。\n\n"
                    f"转写原文：\n{transcript_text}"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload)
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(f"LLM 服务返回 {response.status_code}: {detail}")
        data = response.json()
    return _clean_llm_text(str(data["choices"][0]["message"]["content"]))


async def stream_start() -> str:
    if REALTIME_ASR_PROVIDER == "funasr":
        return await _stream_start_funasr()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{ASR_STREAM_BASE_URL}/api/start")
            if response.status_code >= 400:
                raise RuntimeError(f"流式 ASR 启动失败 {response.status_code}: {response.text}")
            return str(response.json()["session_id"])
    except httpx.TimeoutException as exc:
        raise RuntimeError("流式 ASR 启动超时，可能有旧录音会话正在占用模型，请稍后重试。") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"流式 ASR 服务连接失败: {exc}") from exc


async def stream_chunk(session_id: str, pcm: bytes) -> dict[str, Any]:
    if REALTIME_ASR_PROVIDER == "funasr":
        return await _stream_chunk_funasr(session_id, pcm)
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{ASR_STREAM_BASE_URL}/api/chunk",
                params={"session_id": session_id},
                content=pcm,
                headers={"Content-Type": "application/octet-stream"},
            )
            if response.status_code == 409:
                return {"language": "", "text": "", "segments": []}
            if response.status_code >= 400:
                raise RuntimeError(f"流式 ASR 识别失败 {response.status_code}: {response.text}")
            payload = response.json()
            return {"language": str(payload.get("language") or ""), "text": str(payload.get("text") or ""), "segments": payload.get("segments") or []}
    except httpx.TimeoutException as exc:
        raise RuntimeError("流式 ASR 识别超时。") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"流式 ASR 服务连接失败: {exc}") from exc


async def stream_finish(session_id: str) -> dict[str, Any]:
    if REALTIME_ASR_PROVIDER == "funasr":
        return await _stream_finish_funasr(session_id)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(f"{ASR_STREAM_BASE_URL}/api/finish", params={"session_id": session_id})
            if response.status_code == 409:
                return {"language": "", "text": "", "segments": []}
            if response.status_code >= 400:
                raise RuntimeError(f"流式 ASR 收尾失败 {response.status_code}: {response.text}")
            payload = response.json()
            return {"language": str(payload.get("language") or ""), "text": str(payload.get("text") or ""), "segments": payload.get("segments") or []}
    except httpx.TimeoutException as exc:
        raise RuntimeError("流式 ASR 收尾超时，已释放本地录音状态。") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"流式 ASR 服务连接失败: {exc}") from exc
