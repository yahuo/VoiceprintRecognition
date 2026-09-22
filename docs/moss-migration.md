# MOSS 离线统一链路迁移

## 运行边界

- 实时：FSMN-VAD → Paraformer Streaming → Nano 句末精修；所有请求均使用带 `segmentId/revision/isFinal` 的消息。
- 离线：完整录音 → MOSS（转写、时间、匿名分人）→ CAM++ 身份验证。
- 上传 JSON、上传 SSE、已保存录音 JSON/SSE、会议 CLI 共用 `process_meeting`。
- SSE 保持完整音频输入，通过 vLLM renderer/engine 的 DELTA 输出，在生成期间推送边界闭合且已验证的片段，不等全文生成后集中返回。它不是原生实时音频输入；也不把心跳当作转写流式。正常 EOS 和全文一致性验证后才发送 `done`。
- 会后复核通过已有 `fileId` 入口或网页的“MOSS 会后复核”按钮触发。原始 WAV 保持不变，复核稿显示在独立页签，不覆盖实时稿。不自动写入临床业务系统，也不自动改动人工稿。
- 未选择参会人不运行 CAM++。选定后仍保留全库外部胜者、最短时长、得分与间隔等 guard；短句、重叠或不可信匹配留为未知。匿名标签不等于真实身份。

## 两个 Python 环境，不是两个业务服务

业务环境保留 FunASR/Torch 2.10；MOSS 在本机独立 Python 子进程中运行。没有新增外部语音 API，也不开放 worker 网络端口。通过 stdin + 私有结果 FD 通信，发送的是请求专属的本机 WAV 路径；不使用 pickle/RPC 反序列化。

已验证的 MOSS 栈是 **Linux ARM64、Python 3.11、GB10、vLLM 0.27.1、Torch 2.13.0+cu130、Transformers 5.17.0、FlashInfer 0.6.16.post4**。不要将它装进当前业务 venv，也不能将 Linux venv 搬到 macOS。CUDA/vLLM 在 CPU/MPS 上不回退。

在目标 Linux/CUDA 主机或与实际运行镜像相同的 Python ABI 容器中，显式准备新的环境：

```bash
bash scripts/setup_moss.sh /absolute/path/to/new/venv_moss
python scripts/download_all_models.py --include-moss
```

安装脚本拒绝覆盖已存在的环境。它先按 vLLM 依赖安装，再显式 `--no-deps` 覆盖 FlashInfer post4，解决 post3 的 Python 3.11 导入问题。这是已验证的例外，不是普通无冲突 lockfile；不能直接用未覆盖的 post3 或忽略安装失败。

MOSS 固定下载：

- 仓库：`OpenMOSS-Team/MOSS-Transcribe-Diarize`
- revision：`704aa4a9c304e8520be88901e0d1960158ef5b15`
- 处理器需要 `trust_remote_code`，只从指定的本地快照加载；worker 强制 Hugging Face/Transformers offline。
- 不再需要 Pyannote / HF_TOKEN / 离线 Paraformer / 标点对齐模型。
- 现有声纹索引记录绝对路径；迁移部署应保留原有容器内声纹路径，不能仅移动目录而不迁移索引。不要改动注册向量。

配置示例（不要把真实凭据写进镜像）：

```bash
MOSS_PYTHON=/absolute/path/to/venv_moss/bin/python
MOSS_MODEL_PATH=/absolute/path/to/models/moss/MOSS-Transcribe-Diarize
STREAMING_ASR_MODEL_PATH=/absolute/path/to/models/asr/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
ASR_MODEL_PATH=/absolute/path/to/models/asr/Fun-ASR-Nano-2512
VAD_MODEL_PATH=/absolute/path/to/models/vad/speech_fsmn_vad_zh-cn-16k-common-pytorch
SPK_MODEL_PATH=/absolute/path/to/models/spk/speech_campplus_sv_zh-cn_16k-common
```

在已验证的 cu13 wheel 环境中，`CUDA_HOME` 指向 MOSS venv 的 `lib/python3.11/site-packages/nvidia/cu13`。容器中需要可写的 HF 模块缓存和编译缓存，例如 `HF_HOME=/tmp/hf`、`HF_MODULES_CACHE=/tmp/hf/modules`。不要将缓存指向只读根目录。worker 自身设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，不启用 unsafe pickle。

Compose 提供 `/opt/moss` 与模型只读挂载，但 **必须先准备兼容该镜像 Python、CUDA、架构的 venv**，再设置 `MOSS_PYTHON=/opt/moss/bin/python`。历史 CUDA 12.1 / Python 3.10 Docker 配方不是已验证的 GB10/cu13 制品，不能直接视为迁移完成。源码改造本身不会修改运行中的容器、Pod 或默认服务；构建、镜像发布、部署切换须另行验收与授权。2026-09-21 的 xfusion 更新已获得单独授权，使用同 ABI 的干净新镜像，详见 [xfusion 部署说明](xfusion-deployment.md)。

## 资源与失败策略

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `MOSS_MAX_AUDIO_SECONDS` | 2600 | 覆盖完整 43m11s 验证录音；超长拒绝，不静默截断/独立切块 |
| `MAX_UPLOAD_BYTES` | 268435456 | 单个上传最大 256 MiB |
| `MOSS_START_TIMEOUT_SECONDS` | 600 | worker 冷启动/编译时间上限 |
| `MOSS_TIMEOUT_SECONDS` | 1800 | 单次全文生成时间上限 |
| `MOSS_CANCEL_TIMEOUT_SECONDS` | 30 | 已加载 worker 取消确认上限；超时才硬回收 |
| `MOSS_GPU_MEMORY_UTILIZATION` | 0.25 | vLLM 配置预算，不是实际峰值或硬隔离 |
| `MOSS_CPU_THREADS` | 4 | worker Torch CPU 线程数 |

使用 **单个业务进程**。每个业务进程一个懒加载常驻 MOSS worker，离线只允许一个处理中请求，其余返回 busy；不要用多个 Uvicorn worker 隐式创建多个大模型副本。实时模型的会话缓存独立，模型调用串行保护，可与离线 worker 共存，但同设备并发时的延迟/SLA 尚需压测。

vLLM 固定 BF16、greedy、65,536 最大输出 token、100,000 上下文、单请求、4096-token chunked prefill、禁用 prefix/processor cache、CUDA Graph。chunked prefill 不切断整段音频上下文。增加时长上限不能绕过 token/显存限制，应重新做完整录音验证。

- EOS 未正常结束、输出到 token 上限、非法/倒序时间、不可解析文本：返回失败，不能包装成完整成功。
- 保留合理的跨说话人重叠；只在提取 CAM++ 时避开跨人重叠，不篡改 MOSS 原时间。
- 身份逐片段验证，不依据同簇编号或上一句强行继承姓名，未知不会伪造 0.99/1.0 置信度。流式模式等待后续起点水位越过当前段末，防止尚未生成的跨人重叠污染 CAM++ 取样；匿名模式不需这层等待。
- 缺配置：503；busy：503；时长/体积超限：413；不可解码：422；模型输出异常：502；推理超时：504。
- SSE 已发响应头后用 `error` 事件报告失败，绝不随后发送 `done`。
- 离线 SSE 连接与后台任务分离：刷新/关闭页面不取消推理；按 jobId 恢复已有结果及后续增量，参见 [任务接口](api.md#后台任务与刷新恢复)。任务内存缓存有数量/大小/保留期上限，不提供进程重启后的断点续推。
- 仅显式取消任务或服务关闭才通知推理线程。正常取消通过私有控制通道 abort 当前 vLLM 请求并消费终态，保留模型；若取消发生于首次加载中，会等待该次加载结束、跳过推理并保留已加载模型。取消确认超时、模型/协议异常仍回收自有 worker。任务退出后才删除其临时输入，正常停服关闭常驻模型。
- 原始录音异常断连不提交；显式 stop 后，即使句末识别失败仍保存 WAV 并报告 `transcriptionStatus=failed`。
- FFmpeg 解码限定本机文件/音频容器、16kHz mono PCM16、120秒解码超时；不跟随网络引用或播放列表。先下混/限长再读数组，取消会回收解码子进程。
- 有损格式在不同 FFmpeg 版本上不保证 PCM 逐位相同。实际请求应绑定源文件和本环境 PCM 哈希；标准 16kHz PCM16 WAV 的数据必须逐位保持。不能将不同解码结果伪称同输入 A/B。
- `MAX_UPLOAD_BYTES` 是应用处理上限；生产网关仍需设置请求体上限。JSON 长请求应配置足够的读取超时，SSE 应关闭代理缓冲。

## 接口迁移

`priority=speed|accuracy` 在离线接口保留输入兼容与非法值校验，但不再切换模型或识别策略：

```json
{
  "requestedPriority": "accuracy",
  "effectivePriority": "accuracy",
  "processingMode": "unified",
  "fallbackReason": null,
  "priorityDeprecated": true,
  "method": "moss"
}
```

为兼容严格枚举旧返回值的客户端，`requestedPriority/effectivePriority` 都回显规范化后的请求值，仅供过渡展示，不代表实际处理策略。新客户端用 `processingMode=unified` 与 `method=moss` 判断链路。基本转写字段保持不变，增加 `diarizationSpeaker`。没有旧的 812.8 秒 accuracy 降级，也没有隐藏的 Nano/Pyannote 回退。

实时的 `priority` 仍只是同一流水线的断句/热词策略：默认 12s 连续段上限；accuracy 为 60s、较长尾静音和已有小词表，不增加另一个后端。

## 验证与发布门禁

```bash
python -m unittest discover -s tests -v
```

单测覆盖统一 JSON/SSE、弃用参数、候选外胜者拒识、未知身份、重叠取样、完整 PCM、缓存隔离、尾段排空、推理失败、超限及取消清理；另验证刷新不取消、同任务回放不重复推理、事件游标/任务缓存边界、取消后同 worker 复用及真正异常的回收。网页在发送摘要前确认人工核对与文本发送目的地，摘要提示词禁止自行改写医学事实；这不是对 LLM 输出准确性的保证。实际模型验证须另有证据，mock 测试不是 ASR 精度测试。

既有完整 MDT 的 904/914 段原始输出可用于验证新解析器无文本/时间损失，不需为纯解析改动重复推理。运行栈、提示词或模型变更时再做实际整段回归。中文医学准确率必须有与 PCM 哈希绑定的人工参考；模型差异和匿名标签数不能作为 CER/DER。上线前还需人工复核数字、单位、否定词、术语与身份归属。
