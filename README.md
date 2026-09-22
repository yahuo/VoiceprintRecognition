# 声纹识别会议系统

自托管的会议转写与声纹识别：实时快速出稿，会后完整上下文复核。

| 场景 | 唯一处理链路 |
|---|---|
| 实时麦克风 / WebSocket | FSMN-VAD → Paraformer Streaming → Fun-ASR-Nano 句末精修 |
| 上传 / 已保存录音 / 会议 CLI | MOSS-Transcribe-Diarize（文字、时间、匿名分人） |
| 注册身份验证 | CAM++，保留候选限制与未知拒识 |
| 会议摘要 | 现有 OpenAI-compatible LLM 模块，按配置调用 |

**MOSS 的匿名标签不是注册身份；医学数字、术语、否定词与说话人归属需要人工核对。** 不用 LLM 的医学常识静默改写识别结果。

## 快速开始

### 业务与实时环境

需要 Python 3.10+、FFmpeg；麦克风 CLI 需要 PortAudio/PyAudio。MOSS 独立环境已验证 Python 3.11，见下节。

```bash
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/download_all_models.py
python -m app.server --device cuda:0
# CPU/MPS 可运行实时/CAM++；不提供 MOSS 长录音 CPU/MPS 回退
```

打开 `http://localhost:8000/client`。实时稿会按 `segmentId/revision/isFinal` 原位更新；显式停止并收到 `recording_saved` 后可下载原始 WAV，或点击 **MOSS 会后复核**。复核稿显示在另一页签，不覆盖实时稿。

### MOSS 离线环境（必配，否则离线请求明确失败）

MOSS 的 vLLM/Torch 与业务依赖隔离。在目标 Linux/CUDA 主机或匹配的容器环境中显式准备：

```bash
bash scripts/setup_moss.sh /absolute/path/to/new/venv_moss
python scripts/download_all_models.py --include-moss

export MOSS_PYTHON=/absolute/path/to/new/venv_moss/bin/python
export MOSS_MODEL_PATH=/absolute/path/to/models/moss/MOSS-Transcribe-Diarize
python -m app.server --device cuda:0
```

已验证栈：GB10 / Linux ARM64 / Python 3.11 / vLLM 0.27.1 / Torch 2.13.0+cu130 / Transformers 5.17.0 / FlashInfer post4。模型固定 revision `704aa4a9c304e8520be88901e0d1960158ef5b15`。不要把 MOSS 依赖装入业务 venv。

worker 在本机懒加载、常驻、单请求串行；不暴露网络端口，不上传第三方语音服务。默认完整音频上限 2600 秒，覆盖已验证的43分钟录音；超过上限、输出截断或解析异常均报错，不偷偷切块或降级。

旧 FunASR/Pyannote 镜像升级见 **[简明升级指南](docs/upgrade-to-moss.md)**：准备资源 → 备份 → 改配置 → 启动验收 → 回滚。

运行边界详见 **[MOSS 迁移与运行说明](docs/moss-migration.md)**，包括缓存、CUDA_HOME、超时、资源预算、API 兼容边界和生产切换门禁。

## 运行时配置

```bash
# 实时及声纹模型，可使用 scripts/download_all_models.py 输出的本地路径
ASR_MODEL_PATH=/path/to/models/asr/Fun-ASR-Nano-2512
STREAMING_ASR_MODEL_PATH=/path/to/models/asr/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
VAD_MODEL_PATH=/path/to/models/vad/speech_fsmn_vad_zh-cn-16k-common-pytorch
SPK_MODEL_PATH=/path/to/models/spk/speech_campplus_sv_zh-cn_16k-common

# 独立 MOSS worker
MOSS_PYTHON=/path/to/venv_moss/bin/python
MOSS_MODEL_PATH=/path/to/models/moss/MOSS-Transcribe-Diarize
MOSS_MAX_AUDIO_SECONDS=2600
MOSS_START_TIMEOUT_SECONDS=600
MOSS_TIMEOUT_SECONDS=1800
MOSS_GPU_MEMORY_UTILIZATION=0.25

# 私有数据（默认在项目目录下）
VOICEPRINT_DB_DIR=/path/to/voiceprint_db
RECORDINGS_DIR=/path/to/recordings
```

不再使用 `UPLOAD_ASR_BACKEND`、`UPLOAD_ASR_MODEL_PATH`、`PUNC_MODEL_PATH`、`LIVE_PIPELINE`、`ASR_BACKEND` 或 `HF_TOKEN` 为离线/实时链路选路。服务不会自动加载 Pyannote、离线 Paraformer、Qwen ASR 或其他候选后端。

`GET /` 仅返回安全的链路概要；`offline_configured` 不是 GPU 已就绪的承诺。业务进程使用单 worker，避免多份 MOSS 占用显存。反向代理还应配置上传大小、SSE 超时/禁缓冲及访问控制。

## 声纹注册与候选范围

- 推荐安静环境的单人 10–30 秒音频，16kHz、WAV/M4A/MP3 均可。
- 以不透明 `id` 标识声纹，姓名仅用于展示；同名可以注册多个 id。
- **没有选择参会人时不识别人名**，离线仍会返回 MOSS 的匿名分人。
- 只选一人也必须验证；若库外候选更强或当前片段太短/重叠/低分，保持未知。
- 继续使用原 CAM++ 向量，无需重建声纹库。不会因连续讲话而伪造继承置信度。

```bash
python -m app.utils.voiceprint register --name "医生A" --audio registration.wav
python -m app.utils.voiceprint list
python -m app.utils.voiceprint identify --audio unknown.wav
```

`run.sh register/list/identify/meeting/live` 入口仍可用；`./run.sh delete <声纹ID>` 按 ID 删除声纹，不接受姓名代替 ID。脚本固定使用项目的 `venv`，输入/输出相对路径仍相对于调用目录。已移除会通配删除音频的 `clean` 命令，原始录音不作为临时文件清理。

旧 `scripts/diarization.py`、`scripts/interactive.py` Nano 离线示例已移除；会议转写统一使用以下模块入口。需要指定会议候选时：

```bash
python -m app.services.meeting --audio meeting.wav --output review.md \
  --speaker-id doctor-a --speaker-id doctor-b
python -m app.services.live --device cuda:0 --output live.md --speaker-id doctor-a
```

麦克风 CLI 复用 WebSocket 的流式会话实现，Ctrl+C 排空尾段并保存原始录音。会后对保存的 WAV 运行会议 CLI，输出到新文件。两个 CLI 都拒绝覆盖已有输出，保护实时稿和人工修改。

双音频声纹比对工具仍保留：`python scripts/verification.py --audio1 a.wav --audio2 b.wav`，复用声纹 CLI 的 CAM++ 加载与特征提取，并使用 `SPK_MODEL_PATH` 本地配置。相似度不是身份准确率或概率，阈值需按实际场景校准。

## API

详细契约：[docs/api.md](docs/api.md)。

- 上传 JSON/SSE：`POST /v1/meeting/transcribe[/stream]`
- 已保存录音 JSON/SSE：`POST /v1/meeting/recordings/{fileId}/transcribe[/stream]`
- 实时：`/ws/meeting/live`
- 下载录音：`GET /v1/meeting/recordings/{fileId}`
- 幂等删除：`POST /v1/meeting/recordings/delete`
- 声纹注册/查询/重载/删除：`/v1/voiceprint/*`

离线 `priority=speed|accuracy` 已弃用；旧 `requestedPriority/effectivePriority` 仍兼容回显请求值，避免破坏枚举解析，但不参与模型选路。实际始终返回 `processingMode=unified`、`priorityDeprecated=true`、`method=moss`，不再有两条离线识别实现。

## 会议摘要

`.env` 中的既有配置保持：

```bash
LLM_BASE_URL=https://your-approved-provider/v1
LLM_API_KEY=your-key
LLM_MODEL=your-model
```

语音自托管不等于摘要一定本地：摘要文本流向由该配置决定，需遵守业务的数据授权。网页会在发送前确认人工核对与接收地址；提示词禁止自行改写医学数字或补全事实。摘要仍需人工复核，不写回原始转写。

## 容器与私有数据

- 不将 `.env`、录音、声纹库、实验输出或权重打进应用镜像。
- 音频识别依赖与模型作为运行时资产准备，`voiceprint_db/`、`recordings/` 持久化挂载。
- Compose 已提供 MOSS venv/模型挂载参数；venv 的 Python ABI、CUDA 和架构必须匹配运行镜像。
- **历史 Dockerfile 的 CUDA 12.1/Python 3.10 配方不是已验证的 GB10/cu13 制品。** 不应把源码改造或挂载变量配置完成当成已部署成功；生产镜像构建和切换仍需独立验证。
- 本地源码改动不会重启已有服务，也不自动执行 Compose/Helm 部署。
- xfusion 已另行授权更新：使用 `deploy/Dockerfile.xfusion` 定向构建，参见 [部署、数据保全与回滚说明](docs/xfusion-deployment.md)。

## 代码布局

```text
app/core.py                     模型服务、CAM++ 身份保护、声纹库
app/server.py                   REST/SSE/WebSocket 契约
app/services/moss.py             独立 worker 客户端、输出校验、生命周期
app/services/moss_worker.py      固定 vLLM 全文推理
app/services/meeting.py          离线共享流水线
app/services/streaming.py        FSMN 分段、Paraformer 增量缓存
app/services/live_session.py     唯一实时会话实现
app/services/live.py             麦克风 CLI 适配
app/services/recording_store.py  原始 WAV 存储
static/web_client.html          Web 客户端
```

## 测试与验收

```bash
python -m unittest discover -s tests -v
```

单测不加载 GPU 模型。真实模型、接口传输和完整录音验证另行记录；没有人工参考稿时不报告 CER/DER。性能收益、模型输出差异和声纹自一致性均不能替代医学准确率验收。

参考：[MOSS](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)、[FunASR](https://github.com/modelscope/FunASR)、[Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)。
