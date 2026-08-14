# ASR 统一配置升级说明

本文档用于把实时、上传和离线链路从两套 ASR 配置升级为一套统一配置。

## 变更原则

旧版本同时存在以下重复配置：

- `ASR_BACKEND` 与 `UPLOAD_ASR_BACKEND`
- `ASR_MODEL_PATH` 与 `UPLOAD_ASR_MODEL_PATH`

旧代码还会固定加载 Fun-ASR-Nano，再为上传链路额外加载 Paraformer，导致同类型模型存在两套配置和两个实例。

当前版本只保留：

- `ASR_BACKEND`：实时、上传和离线转写统一使用的后端
- `ASR_MODEL_PATH`：与所选后端对应的唯一 ASR 模型路径

默认后端为 `paraformer`。如设置为 `nano`，所有转写链路都会统一切换到 Fun-ASR-Nano，不再混用两种 ASR 模型。

## 第一步：下载模型

默认下载 Paraformer 以及运行所需的 VAD、PUNC 和 SPK 模型：

```bash
python3 scripts/download_all_models.py
```

如果所有链路需要统一使用 Nano：

```bash
python3 scripts/download_all_models.py --asr-backend nano
```

Paraformer 模式下，离线服务器上的 `models/` 至少应包含：

- `models/asr/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- `models/vad/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- `models/punc/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`
- `models/spk/speech_campplus_sv_zh-cn_16k-common`

Nano 模式只需将 ASR 目录替换为 `models/asr/Fun-ASR-Nano-2512`；Nano 不依赖 Paraformer 模型目录。

## 第二步：修改 `.env`

Paraformer 默认配置：

```bash
ASR_BACKEND=paraformer
ASR_MODEL_PATH=/app/models/asr/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
UPLOAD_ASR_BATCH_SIZE_S=300
MODELS_PATH=./models
VAD_MODEL_PATH=/app/models/vad/speech_fsmn_vad_zh-cn-16k-common-pytorch
PUNC_MODEL_PATH=/app/models/punc/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
SPK_MODEL_PATH=/app/models/spk/speech_campplus_sv_zh-cn_16k-common
```

全局切换到 Nano 时，只修改同一组 ASR 配置：

```bash
ASR_BACKEND=nano
ASR_MODEL_PATH=/app/models/asr/Fun-ASR-Nano-2512
```

请从旧 `.env` 中删除 `UPLOAD_ASR_BACKEND` 和 `UPLOAD_ASR_MODEL_PATH`，它们已不再被代码读取。

## 第三步：重启服务

slim 模式：

```bash
docker compose --profile slim up -d --no-build
```

全量镜像模式：

```bash
docker compose up -d --no-build
```

## 第四步：验证

启动日志应只出现一次 ASR 模型加载过程，并包含所选后端，例如：

```text
加载 ASR 模型 (Paraformer)...
ASR 模型路径: /app/models/asr/...
ASR 模型加载完成 (paraformer)
```

最小验证清单：

1. `.env` 中只有一组 `ASR_BACKEND` 和 `ASR_MODEL_PATH`
2. 容器内只加载所选 ASR 模型
3. 实时 WebSocket、上传和离线转写日志中的 `asr_backend` 一致
4. Paraformer 模式下上传转写仍可生成句子时间戳
