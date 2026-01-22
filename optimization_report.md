# 声纹识别系统优化报告

本文档记录了针对实时会议记录系统（Real-time Meeting Transcription）的关键性能优化与逻辑改进。

## 1. 强制中文 ASR (防幻觉)

**问题**：短语音或模糊发音容易被 Fun-ASR-Nano 误判为日语或韩语（例如 "嗯" 被识别为日语假名）。
**解决方案**：强制指定 ASR 语言为中文 `zh`。

```python
# server.py

# 原代码: language="auto"
# 优化后:
res = self.asr_model.generate(
    input=[audio_path],
    language="zh",  # 强制使用中文，避免短语音误判为日语
    use_itn=True,
    batch_size=1
)
```

## 2. 优化切分粒度 (解决多人混杂)

**问题**：默认的静音检测阈值（1.0秒）过长，导致多人快速对话时会被合并成一段长语音，无法区分说话人。
**解决方案**：将服务端静音切分阈值降至 **0.5秒**。

```python
# server.py

# 原代码: if silence_duration > 1.0 ...
# 优化后:
if silence_duration > 0.5 and len(audio_buffer) > 16000 * 2:
    # 立即切分处理
```

*注：`web_client.html` 中也建议同步调整静音检测参数以获得最佳效果。*

## 3. 说话人继承策略 (解决短语音识别)

**问题**：极短的语音（如"对"、"成熟"、"嗯"）特征过少，声纹识别置信度极低，导致大量 "未知" 说话人。
**解决方案**：实现**上下文继承**。如果当前语音识别为"未知"，但距离上一句有效识别时间很短（<3秒），则认为是对同一人的连续记录。

> [!IMPORTANT]
> 只有识别为"未知"时才触发继承，不能根据置信度判断，否则会错误地把不同人的低分语音都归到同一人。

```python
# server.py

# === 说话人继承策略 ===
current_time = time.time()
# 只有在识别为"未知"时，才考虑继承上一个说话人
# 条件: 识别为未知 + 距离上一句很短(<3秒) + 上一句是已知说话人
if speaker == "未知" and \
   (current_time - last_speech_time < 3.0) and \
   last_speaker != "未知":
    
    speaker = last_speaker
    score = 0.99  # 标记为继承
    print(f"🔄 继承说话人: {last_speaker}")

# 更新会话状态
if speaker != "未知":
    last_speaker = speaker
    last_speech_time = current_time
```

## 4. 低置信度过滤 (去噪)

**问题**：环境噪音或无意义的瞬间声音（<0.2秒）会导致屏幕上出现大量 "未知" 的无意义条目。
**解决方案**：彻底丢弃置信度极低（<0.15）的结果，不再发送给前端。

```python
# server.py

# 过滤置信度极低的结果（通常是音频过短）
MIN_CONFIDENCE = 0.15
if score < MIN_CONFIDENCE:
    # 忽略该结果，不发送给前端
    continue
```

## 5. 声纹匹配阈值微调

**问题**：部分注册用户识别不够灵敏。
**解决方案**：将默认判定阈值从 0.35 下调至 **0.30**。

```python
# server.py

def match_speaker(..., threshold: float = 0.30) -> Tuple[str, float]:
    # ...
```

## 6. 功能增强

- **热重载声纹库**：无需重启服务即可生效新注册的声纹。
  - 接口: `POST /v1/voiceprint/reload`
- **删除声纹接口**：支持通过 API 或脚本删除声纹。
  - 接口: `DELETE /v1/voiceprint/{name}`
  - 脚本: `./run.sh delete "姓名"`
