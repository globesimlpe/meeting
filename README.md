# AI Meeting Transcription

面向 Linux 私有化部署的网页版 AI 会议工具。页面只保留真实可用功能：

- 创建会议后点击开始会议，浏览器麦克风以 16k mono Float32 PCM 推给 Qwen ASR streaming，实时文字直接输出。
- 上传单个长音频或批量音频，调用本地 GPU FunASR 自动 VAD 切段、加标点、生成句级时间戳并识别说话人。
- 基于当前会议文字调用本地 vLLM 生成 AI 纪要总结。

项目不会预置示例会议或模拟转写内容。

## 服务组成

需要同时运行三个核心服务：

```text
8005  Qwen3-ASR-0.6B 统一 ASR 服务，用于实时会议转写
8012  Qwen3.6-35B-A3B vLLM 服务，用于 AI 纪要
GPU   FunASR 离线组件，用于长音频/批量上传转写
8001  meeting-ai 后端 API
5173  meeting-ai 前端开发服务
```

已有 LLM 启动脚本在 `/home/regchen/Chuyi/model_run`：

```bash
/home/regchen/Chuyi/model_run/run_qwen36_B35_A3B_vllm.sh
```

统一 ASR 启动脚本：

```bash
cd /home/regchen/Chuyi/meeting-ai-mvp/backend
./start_streaming_asr.sh
```

## 端口配置

前后端应用端口统一放在 `config/ports.env`：

```bash
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=5173
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8001
API_PROXY_HOST=127.0.0.1
```

`frontend/vite.config.ts` 会读取该文件，把 `/api` 代理到 `http://API_PROXY_HOST:BACKEND_PORT`。`backend/start.sh` 和 `frontend/start.sh` 也会读取同一个配置。模型服务端口仍在 `backend/.env` 和 `/home/regchen/Chuyi/model_run` 对应脚本中管理。

## 后端配置

`backend/.env`：

```bash
LLM_BASE_URL=http://127.0.0.1:8012/v1
LLM_MODEL=Qwen3.6-35B-A3B-NVFP4
LLM_API_KEY=EMPTY

ASR_BASE_URL=http://127.0.0.1:8005
ASR_MODEL=/home/regchen/Chuyi/models/Qwen3-ASR-0.6B
ASR_API_KEY=EMPTY
ASR_TRANSCRIBE_PATH=/v1/audio/transcriptions
ASR_MAX_RETRIES=2

ASR_STREAM_BASE_URL=http://127.0.0.1:8005
ASR_STREAM_MODEL=/home/regchen/Chuyi/models/Qwen3-ASR-0.6B
ASR_STREAM_GPU_MEMORY_UTILIZATION=0.12
ASR_STREAM_MAX_MODEL_LEN=4096
ASR_STREAM_CHUNK_SIZE_SEC=1.0
ASR_STREAM_UNFIXED_CHUNK_NUM=4
ASR_STREAM_UNFIXED_TOKEN_NUM=5

OFFLINE_ASR_PROVIDER=funasr
FUNASR_MODEL=paraformer-zh
FUNASR_VAD_MODEL=fsmn-vad
FUNASR_PUNC_MODEL=ct-punc
FUNASR_SPK_MODEL=cam++
FUNASR_DEVICE=cuda:0
FUNASR_BATCH_SIZE_S=300
FUNASR_MERGE_LENGTH_S=15

REALTIME_ASR_PROVIDER=qwen
REALTIME_FUNASR_INTERVAL_SEC=3.0
REALTIME_FUNASR_MIN_AUDIO_SEC=1.5
REALTIME_FUNASR_TMP_DIR=/tmp/meeting-ai-funasr-stream
```

如果未配置对应服务地址，接口会返回错误，不会生成 mock 文本。

实时转写默认使用 Qwen ASR streaming；FunASR 保留给上传长音频和批量音频做离线切段、时间戳、标点和说话人识别。


FunASR/GPU 依赖在 `meeting-ai` 环境中安装：

```bash
conda activate meeting-ai
python -m pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r backend/requirements.txt
```

## 开发启动

启动统一 ASR：

```bash
cd /home/regchen/Chuyi/meeting-ai-mvp/backend
./start_streaming_asr.sh
```

启动 vLLM 纪要模型：

```bash
/home/regchen/Chuyi/model_run/run_qwen36_B35_A3B_vllm.sh
```

启动应用后端，使用 `meeting-ai` conda 环境并读取 `config/ports.env`：

```bash
cd /home/regchen/Chuyi/meeting-ai-mvp/backend
./start.sh
```

启动前端，同样读取 `config/ports.env`：

```bash
cd /home/regchen/Chuyi/meeting-ai-mvp/frontend
./start.sh
```

访问：

```text
http://服务器IP:5173
```

## 内网麦克风访问

浏览器默认只允许 HTTPS、`localhost`、`127.0.0.1` 页面调用麦克风。公司内网用户直接访问 `http://服务器IP:5173` 时，上传音频可用，但实时录音通常会被浏览器拦截。

推荐给内网地址配置 HTTPS。开发临时方案可以让公司 IT 对 Chrome/Edge 下发企业策略，把当前地址加入允许列表：

```json
{
  "UnsafelyTreatInsecureOriginAsSecure": ["http://服务器IP:5173"]
}
```

策略下发后，用户需要重启浏览器并重新打开页面。

## 当前接口

```text
GET    /api/health
GET    /api/meetings
POST   /api/meetings
GET    /api/meetings/{id}
POST   /api/meetings/{id}/audio
POST   /api/meetings/{id}/audio/batch
POST   /api/meetings/{id}/stream/start
POST   /api/meetings/{id}/stream/chunk
POST   /api/meetings/{id}/stream/finish
POST   /api/meetings/{id}/summary
```
