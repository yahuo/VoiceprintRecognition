# Voiceprint Meeting System API 接口文档

- 版本: `2.0.0`
- Base URL: `http://localhost:8000`
- OpenAPI JSON: [`docs/openapi.json`](openapi.json)

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
- `allowed_speakers`: 可选的参会人白名单。未传时走全库匹配；传入后只在指定注册人范围内识别说话人

请求体:

- 请求体必填: 是
- Content-Type: `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | `string` | 是 | 会议音频文件，支持 WAV、MP3、M4A 等格式。 |
| `threshold` | `number` | 否 | 可选的声纹匹配阈值；不传时使用服务端默认值。 |
| `allowed_speakers` | `array[string] \| null` | 否 | 可选的参会人白名单。可重复传多个同名字段；传入后只会在这些已注册声纹中匹配。 |

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
- `allowed_speakers`: 可选的参会人白名单。未传时走全库匹配；传入后只在指定注册人范围内识别说话人

返回 `text/event-stream`，会按阶段推送 `status / info / segment / done / error` 事件。

请求体:

- 请求体必填: 是
- Content-Type: `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | `string` | 是 | 会议音频文件，支持 WAV、MP3、M4A 等格式。 |
| `threshold` | `number` | 否 | 可选的声纹匹配阈值；不传时使用服务端默认值。 |
| `allowed_speakers` | `array[string] \| null` | 否 | 可选的参会人白名单。可重复传多个同名字段；传入后只会在这些已注册声纹中匹配。 |

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | SSE 流，按阶段和分段持续返回转写结果 | `application/json` |
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
- **file**: 音频文件 (WAV, MP3, M4A 等)

请求体:

- 请求体必填: 是
- Content-Type: `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | `string` | 是 |  |
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

### `DELETE /v1/voiceprint/{name}`
- 摘要: Delete Speaker

删除已注册的声纹

请求参数:

| 名称 | 位置 | 类型 | 必填 | 说明 |
|---|---|---|---|---|
| `name` | `path` | `string` | 是 |  |

响应:

| 状态码 | 说明 | Content-Type |
|---|---|---|
| `200` | Successful Response | `application/json` |
| `422` | Validation Error | `application/json` |

## WebSocket 接口

### `WS /ws/meeting/live`

- 用途: 实时会议识别与录音保存。客户端发送 PCM 音频 bytes，服务端按片段返回 JSON 识别结果；客户端发送停止控制消息后，服务端保存完整录音并返回 `fileId`。
- 连接地址: `ws://localhost:8000/ws/meeting/live`
- Query 参数: `allowed_speakers` 可重复传入，用于限制本次会议的声纹匹配范围。
- 示例: `ws://localhost:8000/ws/meeting/live?allowed_speakers=张三&allowed_speakers=李四`

服务端消息示例:

```json
{
  "time": "14:23:01",
  "speaker": "张三",
  "confidence": 0.82,
  "text": "这里是实时识别出的文本"
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
  "channels": 1
}
```

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
  -F "file=@samples/jinxin.m4a"
curl -X POST http://localhost:8000/v1/meeting/transcribe \
  -F "file=@samples/test_zh.mp3" \
  -F "threshold=0.3" \
  -F "allowed_speakers=张三"
curl -L -o recording.wav \
  http://localhost:8000/v1/meeting/recordings/742d5634-bf12-4384-8099-d85c01858436
curl -X POST http://localhost:8000/v1/meeting/recordings/delete \
  -H "Content-Type: application/json" \
  -d '{"fileIds":["742d5634-bf12-4384-8099-d85c01858436"],"reason":"patient_discharged","requestId":"req-1"}'
```
