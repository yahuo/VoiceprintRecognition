# Voiceprint Meeting System API 接口文档

- 版本: `2.0.0`
- Base URL: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## REST 接口

### `GET /`
- 摘要: Root

健康检查

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | Successful Response | `application/json` |

### `GET /client`
- 摘要: Client

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | Successful Response | `application/json` |

### `POST /v1/meeting/summarize`
- 摘要: Summarize Meeting Api

生成会议总结

Body: { "transcript": [{"speaker": "...", "text": "...", "time": "..."}, ...] }

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | Successful Response | `application/json` |

### `POST /v1/meeting/transcribe`
- 摘要: 上传音频生成会议记录

上传音频文件，返回完整会议记录。

- `file`: 会议音频文件
- `threshold`: 可选的声纹匹配阈值，默认使用服务端配置
- `allowed_speaker_ids`: 可选的参会人声纹 id 白名单。未传时不匹配注册声纹；传入后只在指定注册声纹范围内识别说话人
- `priority`: 可选的识别优先级，默认 `speed`

请求体:

- 请求体必填: 是
- Content-Type: `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | `string` | 是 | 会议音频文件，支持 WAV、MP3、M4A 等格式。 |
| `threshold` | `number` | 否 | 可选的声纹匹配阈值；不传时使用服务端默认值。 |
| `allowed_speaker_ids` | `array[string] \| null` | 否 | 可选的参会人声纹 id 白名单。可重复传多个同名字段；传入后只会在这些已注册声纹中匹配。 |
| `priority` | `string` | 否 | `speed`（默认，分段识别、速度优先）或 `accuracy`（整段上下文、精度优先）。其他值返回 `400`。 |

模式与自动降级规则:

- 不传 `priority` 或传 `priority=speed`: 使用默认的速度优先模式。
- 传 `priority=accuracy`: 优先使用整段上下文识别，并返回一个整段结果。
- 显式请求 `accuracy` 且音频超过 `812.8` 秒: 自动降级为 `speed`，请求不会因整段时长上限失败。

成功响应额外包含以下字段:

| 字段 | 类型 | 说明 |
|---|---|---|
| `requestedPriority` | `string` | 调用方请求的模式。未传 `priority` 时为 `speed`。 |
| `effectivePriority` | `string` | 服务端实际使用的模式。 |
| `fallbackReason` | `string \| null` | 未降级时为 `null`；超过整段上限时为 `accuracy_duration_limit`。 |

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | 完整会议转写结果与 Markdown | `application/json` |
| `422` | Validation Error | `application/json` |

### `POST /v1/meeting/transcribe/stream`
- 摘要: 流式处理会议音频

流式处理会议音频（Server-Sent Events）。

- `file`: 会议音频文件
- `threshold`: 可选的声纹匹配阈值
- `allowed_speaker_ids`: 可选的参会人声纹 id 白名单。未传时不匹配注册声纹；传入后只在指定注册声纹范围内识别说话人
- `priority`: 可选的识别优先级，默认 `speed`；模式和自动降级规则与非 SSE 上传接口一致

返回 `text/event-stream`，会按阶段推送 `status / info / segment / done / error` 事件。
发生自动降级时，服务端会先推送 `type=status, phase=fallback` 事件；最终 `done`
事件包含 `requestedPriority`、`effectivePriority` 和 `fallbackReason`。

请求体:

- 请求体必填: 是
- Content-Type: `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | `string` | 是 | 会议音频文件，支持 WAV、MP3、M4A 等格式。 |
| `threshold` | `number` | 否 | 可选的声纹匹配阈值；不传时使用服务端默认值。 |
| `allowed_speaker_ids` | `array[string] \| null` | 否 | 可选的参会人声纹 id 白名单。可重复传多个同名字段；传入后只会在这些已注册声纹中匹配。 |
| `priority` | `string` | 否 | `speed`（默认）或 `accuracy`。其他值返回 `400`。 |

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | SSE 流，按阶段和分段持续返回转写结果 | `text/event-stream` |
| `422` | Validation Error | `application/json` |

### `GET /v1/voiceprint/list`
- 摘要: List Speakers

列出已注册的声纹

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | Successful Response | `application/json` |

### `POST /v1/voiceprint/register`
- 摘要: Register Speaker

注册声纹

- **name**: 说话人姓名
- **id**: 可选的外部声纹 id；未传时自动生成 24 位 ObjectId 风格随机 id
- **file**: 音频文件 (WAV, MP3, M4A 等)
- 服务启动/读取声纹索引时会自动把旧版 `name -> npy文件` 索引迁移为新版 `id -> {id, name, file}`。

请求体:

- 请求体必填: 是
- Content-Type: `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | `string` | 是 |  |
| `id` | `string` | 否 | 外部声纹 id，任意非空字符串。 |
| `file` | `string` | 是 |  |

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | Successful Response | `application/json` |
| `422` | Validation Error | `application/json` |

### `POST /v1/voiceprint/reload`
- 摘要: Reload Voiceprints

热重载声纹库（无需重启服务）

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | Successful Response | `application/json` |

### `GET /v1/voiceprint/exists`
- 摘要: Voiceprint Exists

按 id 判断声纹是否已注册

请求参数:

| 名称 | 位置 | 类型 | 必填 | 说明 |
|---|---|---|---|---|
| `id` | `query` | `string` | 是 | 声纹 id |

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | Successful Response | `application/json` |
| `422` | Validation Error | `application/json` |

### `DELETE /v1/voiceprint`
- 摘要: Delete Speaker

按 id 删除已注册的声纹

请求参数:

| 名称 | 位置 | 类型 | 必填 | 说明 |
|---|---|---|---|---|
| `id` | `query` | `string` | 是 | 声纹 id |

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | Successful Response | `application/json` |
| `422` | Validation Error | `application/json` |

## WebSocket 接口

### `WS /ws/meeting/live`

- 用途: 实时会议识别与录音保存。客户端发送 PCM 音频 bytes，服务端按片段返回 JSON 识别结果；客户端发送停止控制消息后，服务端保存完整录音并返回 `fileId`。
- 连接地址: `ws://localhost:8000/ws/meeting/live`
- Query 参数: `allowed_speaker_ids` 可重复传入，用于限制本次会议的声纹匹配范围。
- Query 参数: `priority` 可选，支持 `speed`（默认）和 `accuracy`。
- 示例: `ws://localhost:8000/ws/meeting/live?allowed_speaker_ids=speaker-a&priority=accuracy`

实时模式规则:

- `speed`: 保持原有按语音段返回结果的方式，速度优先。
- `accuracy`: 增大同一语音段的上下文，并用相同 `segmentId`、递增 `revision` 和
  `isFinal` 标识临时结果与最终结果；客户端应按 `segmentId` 替换旧文本，而不是追加。
- 实时 WebSocket 不使用离线接口的 `812.8` 秒自动降级规则；单个连续语音段最长
  `60` 秒，达到上限后结束当前段并开始下一段。

服务端消息示例:

```json
{
  "time": "14:23:01",
  "speakerId": "speaker-a",
  "speaker": "张三",
  "confidence": 0.82,
  "text": "这里是实时识别出的文本"
}
```

`accuracy` 可修订消息示例（后续消息使用相同 `segmentId` 和更大的 `revision`）:

```json
{
  "type": "transcript",
  "time": "14:23:01",
  "speakerId": null,
  "speaker": "未知",
  "confidence": 0.0,
  "text": "给一床患者录生命体征",
  "segmentId": "segment-1",
  "revision": 2,
  "isFinal": true
}
```

暂停录音控制消息:

```json
{
  "type": "pause_recording"
}
```

暂停成功消息:

```json
{
  "type": "recording_paused"
}
```

继续录音控制消息:

```json
{
  "type": "resume_recording"
}
```

继续成功消息:

```json
{
  "type": "recording_resumed"
}
```

停止录音控制消息:

```json
{
  "type": "stop_recording"
}
```

录音保存成功消息:

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

`transcriptionStatus` 仅在 `accuracy` 模式返回；最终识别失败时为 `failed`，录音文件仍会保存。

暂停期间客户端不发送 PCM，服务端不录音也不转写。

**最终 WAV 会直接跳过暂停时间段，不写入静音。**

错误消息示例:

```json
{
  "type": "error",
  "message": "以下参会人未注册声纹: 李四"
}
```

### `GET /v1/meeting/recordings/{fileId}`

- 用途: 根据实时录音返回的 `fileId` 下载 WAV 音频文件。
- 成功响应: `audio/wav`
- 下载文件名: `{fileId}.wav`
- `fileId` 不存在返回 `404`
- `fileId` 非法返回 `400`

示例:

```bash
curl -L -o 742d5634-bf12-4384-8099-d85c01858436.wav \
  http://localhost:8000/v1/meeting/recordings/742d5634-bf12-4384-8099-d85c01858436
```

### `POST /v1/meeting/recordings/{fileId}/transcribe`

- 用途: 根据实时录音返回的 `fileId` 直接识别录音内容，不需要客户端重新上传音频。
- 返回结构: 与 `POST /v1/meeting/transcribe` 一致，包含转写结果及 `requestedPriority`、`effectivePriority`、`fallbackReason`。
- 模式与自动降级规则: 与 `POST /v1/meeting/transcribe` 一致。
- `fileId` 不存在返回 `404`
- `fileId` 非法返回 `400`

查询参数:

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `threshold` | `number` | 否 | 可选的声纹匹配阈值；不传时使用服务端默认值。 |
| `allowed_speaker_ids` | `array[string] \| null` | 否 | 可选的参会人声纹 id 白名单。可重复传多个同名查询参数；传入后只会在这些已注册声纹中匹配。 |
| `priority` | `string` | 否 | `speed`（默认）或 `accuracy`。显式请求 `accuracy` 且音频超过 `812.8` 秒时自动降级为 `speed`。 |

示例:

```bash
curl -X POST \
  "http://localhost:8000/v1/meeting/recordings/742d5634-bf12-4384-8099-d85c01858436/transcribe?allowed_speaker_ids=speaker-a&priority=accuracy"
```

### `POST /v1/meeting/recordings/{fileId}/transcribe/stream`

- 用途: 根据实时录音返回的 `fileId` 流式识别录音内容，前端“一键转录”使用该接口。
- 响应: `text/event-stream`
- 事件结构: 与 `POST /v1/meeting/transcribe/stream` 一致，按阶段返回 `status / info / segment / done / error`。
- 模式、自动降级规则和完成事件元数据: 与 `POST /v1/meeting/transcribe/stream` 一致。
- `fileId` 不存在返回 `404`
- `fileId` 非法返回 `400`

查询参数:

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `threshold` | `number` | 否 | 可选的声纹匹配阈值；不传时使用服务端默认值。 |
| `allowed_speaker_ids` | `array[string] \| null` | 否 | 可选的参会人声纹 id 白名单。可重复传多个同名查询参数；传入后只会在这些已注册声纹中匹配。 |
| `priority` | `string` | 否 | `speed`（默认）或 `accuracy`。显式请求 `accuracy` 且音频超过 `812.8` 秒时自动降级为 `speed`。 |

示例:

```bash
curl -N -X POST \
  "http://localhost:8000/v1/meeting/recordings/742d5634-bf12-4384-8099-d85c01858436/transcribe/stream?allowed_speaker_ids=speaker-a&priority=accuracy"
```

### `POST /v1/meeting/recordings/delete`

- 用途: 由业务系统按业务事件主动清理录音文件，例如病人出院后清理相关查房录音。
- 语义: 服务端只按 `fileId` 删除，不保存 `patientId`、住院号等业务字段。
- 幂等: 已删除或不存在的 `fileId` 会进入 `missing`，不会导致整个请求失败。
- `failed` 非空时，响应 `status` 为 `partial`。

请求体:

```json
{
  "fileIds": ["742d5634-bf12-4384-8099-d85c01858436"],
  "reason": "patient_discharged",
  "requestId": "business-request-id"
}
```

响应示例:

```json
{
  "status": "success",
  "deleted": ["742d5634-bf12-4384-8099-d85c01858436"],
  "missing": [],
  "failed": []
}
```

## 常用调用示例

```bash
curl http://localhost:8000/
curl http://localhost:8000/v1/voiceprint/list
curl -X POST http://localhost:8000/v1/voiceprint/reload
curl -X POST http://localhost:8000/v1/voiceprint/register \
  -F "name=张三" \
  -F "id=speaker-a" \
  -F "file=@samples/jinxin.m4a"
curl "http://localhost:8000/v1/voiceprint/exists?id=speaker-a"
curl -X DELETE "http://localhost:8000/v1/voiceprint?id=speaker-a"
curl -X POST http://localhost:8000/v1/meeting/transcribe \
  -F "file=@samples/test_zh.mp3" \
  -F "threshold=0.3" \
  -F "allowed_speaker_ids=speaker-a" \
  -F "priority=accuracy"
curl -L -o recording.wav \
  http://localhost:8000/v1/meeting/recordings/742d5634-bf12-4384-8099-d85c01858436
curl -X POST \
  "http://localhost:8000/v1/meeting/recordings/742d5634-bf12-4384-8099-d85c01858436/transcribe?allowed_speaker_ids=speaker-a&priority=accuracy"
curl -N -X POST \
  "http://localhost:8000/v1/meeting/recordings/742d5634-bf12-4384-8099-d85c01858436/transcribe/stream?allowed_speaker_ids=speaker-a&priority=accuracy"
curl -X POST http://localhost:8000/v1/meeting/recordings/delete \
  -H "Content-Type: application/json" \
  -d '{"fileIds":["742d5634-bf12-4384-8099-d85c01858436"],"reason":"patient_discharged","requestId":"req-1"}'
```
