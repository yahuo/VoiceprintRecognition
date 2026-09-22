# 升级说明

旧 FunASR/Pyannote 版本升级到当前版本，请按 **[MOSS 简明升级指南](docs/upgrade-to-moss.md)** 操作。

当前链路固定为：

- 实时：FSMN-VAD → Paraformer Streaming → Nano 句末精修。
- 上传/会后复核：完整音频 → MOSS → CAM++ 身份验证。

旧 `ASR_BACKEND`、`LIVE_ASR_BACKEND`、`UPLOAD_ASR_BACKEND` 不再切换后端。`ASR_MODEL_PATH` 应指向 Nano，不能继续指向旧的整段 Paraformer；流式 Paraformer 使用独立的 `STREAMING_ASR_MODEL_PATH`。

配置变量见 [.env.example](.env.example)，模型与独立 MOSS 环境的准备方法见 [运行说明](docs/moss-migration.md)。不要按旧的“所有链路共用一个 ASR 模型”步骤重新部署。
