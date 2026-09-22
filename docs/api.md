# Voiceprint Meeting API 3.0

基本地址示例：`http://localhost:8000`。网页：`GET /client`。

语音处理均在自托管环境中运行。实时稿与会后复核稿是不同版本；自动识别结果不是医学事实或身份认证结论。

## 健康状态

`GET /` 返回 `models_loaded`、注册声纹数、固定的实时/离线链路名称和 `offline_configured`。

`offline_configured` 仅表示已提供 MOSS 配置，不代表 worker 已热启动、CUDA 可用或已通过真实音频验收。健康接口不返回 API key、LLM 地址、模型路径等私有配置。

## 声纹管理

### 注册

`POST /v1/voiceprint/register`，`multipart/form-data`：

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | 是 | 展示名，去除首尾空白后不可为空 |
| `file` | 是 | 注册音频，推荐安静环境单人 10–30 秒 |
| `id` | 否 | 不透明外部 id；不传生成 24 位随机 id。同 id 覆盖原记录，同名可有多个 id |

成功返回 `status/id/name/message/embedding_shape`。提取失败返回 400。声纹向量不通过 API 返回。

```bash
curl -X POST http://localhost:8000/v1/voiceprint/register \
  -F 'id=doctor-a' -F 'name=医生A' -F 'file=@registration.wav'
```

### 查询与删除

- `GET /v1/voiceprint/list` → `{status, count, speakers:[{id,name}]}`。
- `GET /v1/voiceprint/exists?id=doctor-a` → `{registered, id, name}`。
- `DELETE /v1/voiceprint?id=doctor-a`：不存在返回 404，成功返回 `{status,id,name,message}`。
- `POST /v1/voiceprint/reload`：重新加载已有索引与向量。

id 不用于构造向量文件路径，旧名字索引仍支持迁移。不要把姓名或录音文件名当作声纹识别结果。

## 离线全文识别

### 四个共用入口

| 接口 | 输入 | 输出 |
|---|---|---|
| `POST /v1/meeting/transcribe` | multipart 上传 `file` | JSON |
| `POST /v1/meeting/transcribe/stream` | multipart 上传 `file` | SSE |
| `POST /v1/meeting/recordings/{fileId}/transcribe` | 已保存的原始 WAV | JSON |
| `POST /v1/meeting/recordings/{fileId}/transcribe/stream` | 已保存的原始 WAV | SSE |

共同参数：上传接口通过 Form，其余通过 Query 传入。

| 参数 | 默认 | 说明 |
|---|---|---|
| `threshold` | 服务配置（0.30） | 0–1 的有限数值；不能绕过身份保护阈值 |
| `allowed_speaker_ids` | 未指定 | 可重复；只匹配选定的已注册声纹 id。未知 id 返回 400 |
| `priority` | `speed` | **已弃用的兼容参数**；接受 `speed/accuracy`，其他值 400；不再改变离线链路 |

未选择参会人时，仅提供 MOSS 匿名分人，不提取声纹、不查全库身份。即使只选一人，也不能把所有片段强制分给此人；存在更强的候选外注册人时仍拒识。

所有离线请求运行 **同一套 MOSS 完整输入推理**，没有旧的 `accuracy` 整段 Nano 或 `speed` 分段 ASR/对齐分支，没有 812.8 秒自动降级。默认最多 2600 秒完整音频、256 MiB 上传文件；超限明确报错，不截断、不以独立分块结果冒充整段处理。

### JSON

```json
{
  "status": "success",
  "segments": 1,
  "transcript": [{
    "time": "00:01",
    "start_ms": 1200,
    "end_ms": 3200,
    "diarizationSpeaker": "S01",
    "speakerId": null,
    "speaker": "陌生人1",
    "confidence": 0.0,
    "text": "示例发言。"
  }],
  "markdown": "# 会议复核稿...",
  "requestedPriority": "speed",
  "effectivePriority": "speed",
  "processingMode": "unified",
  "fallbackReason": null,
  "priorityDeprecated": true,
  "method": "moss"
}
```

- `diarizationSpeaker`：本次完整录音中的匿名标签，不能跨录音当作真实身份。
- `speakerId`：仅通过 CAM++ guard 后才返回注册 id；否则 `null`。
- `confidence`：声纹匹配分数，不是文本准确率、匿名分人准确率或概率；未确认身份为 0。
- 允许跨说话人时间重叠。不会为让时间线“好看”而剪掉原始结果。
- `requestedPriority/effectivePriority` 均兼容回显规范化后的 `speed/accuracy`，避免破坏旧客户端的枚举解析；标记已弃用，不再代表实际识别策略。新客户端看 `processingMode=unified`、`method=moss`。其余原有转写字段保持。

### SSE

`Content-Type: text/event-stream`，每个 JSON 事件以 `data: ...\n\n` 分隔：

1. `status`：`phase=queued/loading/transcribing`，包含 `message`。
2. `segment`：生成过程中逐段推送，`index` 从 0 连续递增，其他字段与 JSON `transcript` 中的条目一致。
3. `info`：正常 EOS 后确定整数 `total_segments`、`method=moss` 及弃用参数元数据。它可能晚于已经发出的 `segment`；少量等待身份验证的尾段可在它之后发出。不要要求先收到总段数才显示文字。
4. `done`：`segments`、`method` 和弃用参数元数据。
5. `error`：失败时返回 `message`，**不会再发送 done**。

输入仍是一次完整录音，不独立切块。MOSS 生成过程中，将格式边界和时间戳已验证的完整片段逐段推送；选定参会人时还需等待后续时间水位排除跨人重叠，才执行 CAM++ 匹配。期间可能有 `: processing` 注释心跳，但心跳不代表文字输出。

只有正常 EOS、全文校验、流式前缀与终稿一致、全部片段发送完成后才发 `done`。此前显示的是尚未完成的转写；若后续截断、取消或异常，已显示的部分不能冒充成功结果。网页在 `done` 前禁用导出和摘要，失败时保留未完成状态。HTTP 200、收到片段、连接关闭均不能替代 `done`。

```bash
curl -N http://localhost:8000/v1/meeting/transcribe/stream \
  -F 'file=@meeting.wav' -F 'allowed_speaker_ids=doctor-a' -F 'allowed_speaker_ids=doctor-b'

curl -X POST \
  'http://localhost:8000/v1/meeting/recordings/742d5634-bf12-4384-8099-d85c01858436/transcribe?allowed_speaker_ids=doctor-a'
```

JSON 错误码：400 参数，404 录音不存在，413 超限，422 音频无法解码，502 模型失败/输出异常，503 未配置或离线忙，504 推理超时。SSE 已发送响应头后的失败使用 `error` 事件。

### 后台任务与刷新恢复

上传 SSE 可传 Form `job_id`，保存录音 SSE 可传 Query `job_id`，均为规范小写 UUIDv4；不传则由服务端生成。网页在上传前将最近的随机 jobId 与原服务器地址存入本机浏览器 `localStorage`，不保存转写正文。同一浏览器刷新或重新打开同一页面地址可恢复该任务。首次 `status` 和响应头 `X-Meeting-Job-Id` 返回任务 ID。

- `GET /v1/meeting/jobs/{jobId}`：状态 `running/cancelling/done/error/cancelled`、源文件显示名、片段/事件数。
- `GET /v1/meeting/jobs/{jobId}/events?after=0`：从头回放再继续订阅；每个 JSON 事件带连续整数 `eventId`。断线重连可以传已应用的最后一个 eventId；刷新后空白页面从 0 重建结果。`index` 仍是从 0 开始的片段序号，不等于 eventId。
- `POST /v1/meeting/jobs/{jobId}/cancel`：显式取消，最终发 `error`（`cancelled=true`），不发 `done`。不会删除已保存原始录音。正常取消只 abort 当前 vLLM 请求，模型继续常驻；无法确认取消或模型/协议异常才回收 worker。
- **关闭网页、刷新、SSE 断开只释放订阅，不取消任务、不卸载模型。** 输入文件由后台任务持有，计算结束后才清理。重复提交相同 jobId 返回 409，恢复使用 GET，不再次 POST 音频。
- 单进程最多一个进行中任务；最多缓存 8 个任务，每个事件缓存上限 32MiB、同时最多 4 个订阅。已完成结果可恢复至完成后 1 小时，容量不足时可能提前淘汰最旧且无人读取的完成稿。过期/未接受/重启丢失返回 404 或 410；不提供可枚举其他任务的列表，jobId 不应分享。
- 仅保证**已经接受的离线任务**跨页面刷新。上传未完成可能需要重新选择文件；服务进程重启不支持推理断点续跑，已保存录音不受影响。实时麦克风仍属于当前页面，刷新前应先停止保存，不能将离线恢复承诺扩展到麦克风录音。

网页恢复绑定原服务器，重新核验连续片段、事件及最终总数，只有 `done` 才启用下载/摘要。短暂网络错误只重连订阅，不能借重连重新推理。

## 实时 WebSocket

`/ws/meeting/live?allowed_speaker_ids=doctor-a&allowed_speaker_ids=doctor-b`

- 输入：16kHz、单声道、PCM16 little-endian 二进制音频；建议小包持续发送，不是 WAV/MP3 文件容器。
- 普通 PCM 包跨采样边界时会缓存半个采样；单包最多 2 MiB。
- 未传参会人只转写、不匹配人名。
- 所有模式使用 FSMN-VAD、Paraformer 增量缓存、Nano 句末精修，不再保留 legacy 能量切分链路。
- `priority=speed` 默认：约 500ms 端点静音、连续段上限12s。
- `priority=accuracy`：同一流水线使用至少1500ms端点静音、连续段上限60s和已有少量热词。这不是另一个 ASR 后端。
- 模型缓存按会话/段隔离；队列有界，可合并过期 interim，不丢弃尚未送入模型的 PCM。

转写消息：

```json
{
  "type": "transcript",
  "segmentId": "segment-1",
  "revision": 2,
  "isFinal": false,
  "time": "10:20:30",
  "start_ms": 500,
  "end_ms": 1700,
  "speakerId": null,
  "speaker": "未知",
  "confidence": 0.0,
  "text": "临时文字"
}
```

同一 `segmentId` 按更高 `revision` 更新，不应追加为重复句子。`isFinal=true` 为该段精修稿，不等于整场会后复核稿或人工确认。身份仅在句末保守验证，未知不继承上一位说话人。

若首遍失败，发送 `type=status, phase=fallback` 并继续句末精修。若句末失败但有首遍文字，发送 `degraded=true` 的 final 保留首遍稿，再发送带段 id 的 `error`；没有首遍也发送最终错误，不假称成功。

### 控制与录音保存

客户端发送 JSON 文本消息：

- `{"type":"pause_recording"}`：排空当前段，服务端回复 `recording_paused`；暂停期间音频不入录音。
- `{"type":"resume_recording"}`：恢复，回复 `recording_resumed`；时间戳按实际保存 PCM 连续计时。
- `{"type":"stop_recording"}`：排空尾段、等待精修、提交原始 WAV，回复：

```json
{
  "type": "recording_saved",
  "fileId": "742d5634-bf12-4384-8099-d85c01858436",
  "format": "wav",
  "sampleRate": 16000,
  "channels": 1,
  "transcriptionStatus": "complete"
}
```

`transcriptionStatus` **所有模式都会返回**；句末失败时为 `failed`，原始录音仍可用于会后 MOSS 重试。异常断连不会提交录音；客户端应等待 `recording_saved` 再关闭连接。保存的只是原始录音，转写稿由调用方保存；网页保留实时稿与复核稿的独立状态，不实现病历持久化/人工稿编辑。

### 原始录音管理

- `GET /v1/meeting/recordings/{fileId}`：下载 `{fileId}.wav`。
- `POST /v1/meeting/recordings/delete`，JSON：`{"fileIds":["UUID"],"reason":"可选","requestId":"可选"}`。
- 返回 `{status, deleted, missing, failed}`，幂等删除，部分失败时 `status=partial`。
- fileId 必须是 UUID；服务端不保存病人 id/住院号等业务关联。

## 会议摘要

`POST /v1/meeting/summarize`，JSON：

```json
{"transcript":[{"speaker":"医生A","time":"00:01","text":"示例文字"}]}
```

仍使用现有 OpenAI-compatible 摘要配置，可选 `base_url/api_key/model` 覆盖参数保持兼容。语音本地处理不代表摘要一定本地：文本是否发送外部 LLM 取决于该配置与业务授权。建议人工核对后再生成摘要，不把 LLM 医学常识当作修改原文数字的依据。
