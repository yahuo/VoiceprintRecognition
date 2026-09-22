# 旧 FunASR/Pyannote → MOSS 升级简明指南

**不是只换镜像：还需要独立 MOSS 环境和模型。原声纹无需重新录制。**

以下按现有 Docker Compose 部署说明，服务名以 `voiceprint-server` 为例，请按实际替换。

## 1. 备齐新版资源

- 新业务镜像、与镜像兼容的 MOSS 环境。
- Nano、流式 Paraformer、VAD、CAM++ 和 MOSS 模型。

目前已验证的是 **Linux ARM64 / GB10**，镜像未推送仓库，需取得匹配的镜像包后离线导入：

```bash
docker load -i 新版镜像.tar
```

其他架构不能直接复用该镜像。环境和模型的准备方法见 [MOSS 运行说明](moss-migration.md)；资源未就绪，不要先停旧服务。

## 2. 停写并备份

暂停新请求，等正在录音、转写的任务结束，保存需要的结果，再停服务：

```bash
docker compose stop --timeout 120 voiceprint-server
```

保留旧镜像、Compose 配置和 `.env`，备份完整声纹库及录音目录。**新版使用数据副本，原目录留作回滚**，因为旧声纹索引可能自动迁移。

## 3. 修改现有 Compose 配置

沿用原端口、网络和单实例部署，只调整以下内容：

| 配置 | 修改内容 |
|---|---|
| `image` | 换成新镜像的固定版本，不覆盖旧标签 |
| 声纹、录音挂载 | 宿主目录改为副本；容器内路径保持原样 |
| `VOICEPRINT_DB_DIR`、`RECORDINGS_DIR` | 显式设置为上述容器内路径 |
| 模型、MOSS 环境挂载 | 挂载已准备好的目录，模型和运行环境只读 |
| `MOSS_PYTHON` | 容器内独立 MOSS 环境的 `bin/python` 路径 |
| `MOSS_MODEL_PATH` | 容器内 MOSS 模型目录 |
| `STREAMING_ASR_MODEL_PATH` | 容器内流式 Paraformer 模型目录 |

确认 Nano/VAD/CAM++ 路径仍有效，CUDA 与可写缓存配置符合 [运行说明](moss-migration.md)。不要将旧业务 venv 当作 MOSS 环境，也不要直接套用旧 Dockerfile 重新构建。

## 4. 启动并验收

```bash
docker compose config --quiet
docker compose up -d --no-build --pull never voiceprint-server
docker compose logs --tail 50 voiceprint-server
```

打开原网页，确认：

- 声纹列表正常，原录音能下载。
- 实时录音可以识别、停止并保存。
- 上传短音频有增量片段和正常 `done`；刷新能恢复同一任务。

**不能只看健康接口或 HTTP 200。** 验收通过后恢复业务入口。

## 5. 失败时回滚

停掉新版，恢复旧镜像、旧配置（含 `.env`）和原数据目录，再启动旧服务。不要把迁移后的索引直接交给旧镜像。

如果新版已经产生新声纹或录音，先另外保存这些数据，**不要直接用旧备份覆盖**。旧镜像和原始数据先保留，不在升级时清理。
