# 测试

## 不加载模型的回归

```bash
python -m unittest discover -s tests -v
node --test tests/web_client.test.cjs
```

Python 覆盖身份保护、统一离线入口、真实 ASGI 路由、严格 MOSS 解析、worker 生命周期、流式 PCM/缓存、停止排空、录音存储和异常清理。worker 协议用独立的合成子进程验证，不下载模型。新增门禁以“收到真实片段后才允许生产者结束”的同步握手证明增量输出，并覆盖未来重叠水位、增量不重置总超时、部分输出后失败/取消及最终前缀一致性。

任务生命周期回归覆盖 ASGI 断连后继续推理、重连回放不重复计算、显式取消/停服后输入清理、保存录音不删除、有界事件缓存/订阅数/保留期，以及加载中、生成中和 EOS 发布时取消后的同 worker 复用。真正的超时/协议错误仍必须回收进程。

配置回归覆盖 Nano/流式 Paraformer/VAD/CAM++ 的独立路径、`load_live=False` 与 MOSS 懒加载；旧 ASR 后端开关不恢复旧链路，`.env.example` 路径须与模型下载器一致。镜像内运行这些测试时，需额外只读挂载 `.env.example` 和 `scripts/download_all_models.py` 到项目对应位置；不用真实 `.env`，也不把测试输入打进业务镜像。

工具与注册回归覆盖 `run.sh` 的项目 venv/跨目录调用、按不透明 ID 删除、缺参/失败退出，以及拒绝旧 `clean` 时保留音频和未知临时文件；脚本测试在临时目录使用假解释器，不调用真实删除/注册。声纹测试覆盖 CPU 张量与模拟设备张量转换、批次维度、零向量、共享比对工具、实际 ASGI 注册上传大小限制和失败清理，不代替 GPU 模型验收。

Node 使用内置 runner 和隔离 DOM/网络 stub，检查逐段修订、不覆盖实时稿、输出转义、SSE 完成条件、刷新/重新打开时按原服务器恢复、游标续传及取消与 done 竞态；不等于浏览器视觉或真实麦克风验收。

## 真实 `transcribe/stream` 接口

需要已运行且配置好 `MOSS_PYTHON` / `MOSS_MODEL_PATH` 的测试服务，并显式提供音频：

```bash
STREAM_TEST_AUDIO=/absolute/path/to/sample.wav \
STREAM_TEST_BASE_URL=http://127.0.0.1:8000 \
python -m unittest -v tests.test_transcribe_stream_api
```

未设置音频时该项跳过。只能使用授权的测试服务与音频，不要指向第三方或生产服务做无意写入/昂贵推理。

输出首个 SSE 事件、首个真实片段、片段到达跨度、总耗时和片段数。**首个事件通常是进度，不是识别文字**。MOSS 输入保持完整，在生成期间逐段输出；必须另测首段等待和持续输出，不能把终点集中刷出的片段当作流式体验。总耗时除以片段数不是单段模型耗时。

可选门限：

```bash
STREAM_TEST_MAX_FIRST_EVENT_SECONDS=25
STREAM_TEST_MAX_FIRST_SEGMENT_SECONDS=60
STREAM_TEST_MIN_SEGMENT_SPAN_SECONDS=5
STREAM_TEST_MAX_TOTAL_SECONDS=1800
```

根据音频时长、实际硬件、冷/热启动条件设置门限，不能沿用旧离线 Paraformer 的短片段假设；单句音频不适合验证持续输出跨度。参见 [迁移说明](../docs/moss-migration.md)。

## 人工与实际模型验收

- 完整源音频/PCM 哈希一致，不用截断、独立分块结果冒充全文验收。
- JSON/SSE 文字、时间、匿名标签一致，正常 EOS；SSE 必须收到 `done` 且没有 `error`。
- 在真实生成中断开订阅、以同一 jobId 继续和回放，确认只推理一次、片段不丢不重；显式取消不产生 done，取消后的新任务复用相同 worker/EngineCore PID。冷启动取消和热模型取消分开测量。
- 实时以 1× 独立收发，验证 stop 尾段、revision、最终状态、保存 WAV 的逐字节 PCM 一致性。
- 医学数字、术语、单位、否定词和身份需要独立人工参考。机器差异、标签数量或 CAM++ 自一致性不是 CER/DER。
- 私有录音、转写、声纹和实验结果不提交 Git、不放进镜像。
