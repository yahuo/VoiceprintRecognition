# 离线模式升级说明

本文档用于把原先的离线模式升级到当前的离线模式。

## 旧模式与新模式的区别

原先的离线模式主要使用：

- `Fun-ASR-Nano-2512`
- `speech_fsmn_vad_zh-cn-16k-common-pytorch`
- `speech_campplus_sv_zh-cn_16k-common`

当前的离线模式默认改为：

- 上传/离线转写：`Paraformer`
- 实时链路：`Fun-ASR-Nano-2512`
- 说话人分离：`Pyannote`
- 说话人验证：`CAM++`

因此，相比旧模式，需要额外准备：

- `Paraformer` 上传/离线 ASR 模型
- `ct-punc` 标点模型

说明：

- `pyannote` 不需要额外下载，仓库已内置 `models/pyannote`
- 原有的 `VAD`、`SPK`、`Nano` 模型仍然继续使用

## 第一步：下载额外模型

在有网机器的项目根目录执行：

```bash
python3 scripts/download_all_models.py --include-upload-asr
```

该脚本会在现有 `models/` 目录中补齐以下模型：

- `models/asr/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- `models/punc/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`

如果旧模型已存在，脚本会自动跳过，不会重复下载。

## 第二步：同步模型目录到离线服务器

升级后，离线服务器上的 `models/` 至少应包含：

- `models/asr/Fun-ASR-Nano-2512`
- `models/asr/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- `models/vad/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- `models/punc/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`
- `models/spk/speech_campplus_sv_zh-cn_16k-common`

## 第三步：修改 `.env`

如果使用最新的 `docker-compose.yml`，建议在 `.env` 中明确配置：

```bash
UPLOAD_ASR_BACKEND=paraformer
MODELS_PATH=./models
ASR_MODEL_PATH=/app/models/asr/Fun-ASR-Nano-2512
UPLOAD_ASR_MODEL_PATH=/app/models/asr/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
VAD_MODEL_PATH=/app/models/vad/speech_fsmn_vad_zh-cn-16k-common-pytorch
PUNC_MODEL_PATH=/app/models/punc/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
SPK_MODEL_PATH=/app/models/spk/speech_campplus_sv_zh-cn_16k-common
```

其中关键新增项为：

- `UPLOAD_ASR_BACKEND=paraformer`
- `UPLOAD_ASR_MODEL_PATH=...`
- `PUNC_MODEL_PATH=...`

原有的 `VAD_MODEL_PATH` 如果已经配置，可继续复用。

## 第四步：重启服务

### slim 模式

```bash
docker compose --profile slim up -d --no-build
```

### 全量镜像模式

```bash
docker compose up -d --no-build
```

## 第五步：验证升级是否生效

启动后检查日志，确认出现类似信息：

- `上传 ASR 模型加载完成`
- `上传 ASR 复用 VAD 模型目录`
- `上传 ASR 复用 PUNC 模型目录`

如果这些日志都出现，说明离线模式已经切换到当前实现。

## 最小升级清单

1. 执行 `python3 scripts/download_all_models.py --include-upload-asr`
2. 把更新后的 `models/` 同步到离线服务器
3. 在 `.env` 中增加 `UPLOAD_ASR_BACKEND`、`UPLOAD_ASR_MODEL_PATH`、`PUNC_MODEL_PATH`
4. 重启 `docker compose`
5. 检查启动日志确认 Paraformer / VAD / PUNC 已加载
