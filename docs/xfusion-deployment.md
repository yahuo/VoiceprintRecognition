# xfusion 定向部署说明

## 2026-09-21 更新方式

此次按用户授权更新现有 Rancher 开发实例，沿用原 Deployment、Service、Nginx 路径与 PVC，不新建集群/网关，不接管 Helm。

- 业务服务 v3.0.0：实时 Paraformer Streaming + Nano，离线 MOSS + CAM++。
- `deploy/Dockerfile.xfusion` 从干净 Python 3.11.16 / Debian bookworm 基础层构建，只从已验收依赖镜像复制 `/usr/local`。**不继承旧应用层或 Pyannote 权重层**。
- 构建上下文白名单仅包含应用、网页、所需 Fun-ASR 源码及源文件 SHA256 清单。凭据、模型、录音、声纹库和实验输出均不在镜像中。
- 本次目标仅 `linux/arm64` / GB10；不是多架构发布声明。镜像在目标机本地构建，离线导入现有 containerd，按摘要部署，没有推送 registry。
- 业务依赖快照为 Torch 2.10.0+cu130、FunASR 1.3.13、Transformers 5.17.0；MOSS 继续使用独立的 vLLM 0.27.1 / Torch 2.13.0+cu130 / FlashInfer post4 环境。

MOSS 运行环境与模型按文件哈希核对后复制进 PVC 的独立版本目录，不依赖实验目录继续存在。保持测试过的容器内路径 `/work/model`、`/work/optimization/venv-vllm` 及 Python ABI；**不是在任意操作系统之间搬运 venv，也不是一次全新 pip 安装验证**。

## 同日增量 SSE 修复

2026-09-21 发布 `20260921-xfusion-sse`，基于已验证的 UI 镜像仅更新 MOSS 输出链路、完整片段解析/身份水位和网页。模型、完整输入方式、运行依赖与资源配置未变。

- 同机隔离新镜像使用完整 2590.903 秒 MDT：预热后首段 **9.04 秒**，总耗时 **812.04 秒**，914 段持续到达（总耗时一半前已到 519 段）。文字、时间与匿名标签逐项等于原终稿；不代表 CER/DER 验收或总计算提速。
- 正式 HTTPS 入口的 103.032 秒录音：选定参会人时首段 **0.55 秒**、总计 **8.87 秒**，42 段、正常 `done`。这不是浏览器上传耗时。
- 真实 Chrome 经 SSH 转发上传完整同一音频，包含上传/网络后的首段显示 **7.53 秒**、完成 **15.43 秒**；观察到 41 次未完成状态的增量 DOM 更新，期间导出/摘要保持禁用，完成后才启用。未调用摘要服务。
- 仅首段与持续显示体验改善；完整输入仍须先上传，首次启动 worker 的模型加载还需额外时间。正常 EOS/全文一致性检查后才确认完成，失败保留“未完成”状态。

## 持久化与资源

- 原声纹路径 `/data/voiceprints`、录音路径 `/data/recordings` 不变。保留原向量与索引；更新前备份，更新后核对哈希。
- 只读挂载业务模型、MOSS 模型与 venv；HF 临时缓存位于可写 `/tmp`，编译缓存单独持久化。
- 沿用现场 root UID，以保持原文件权限及回滚兼容；业务容器 `capabilities.drop=ALL`、禁止提权、只读根、不挂 ServiceAccount token、不使用 hostNetwork/privileged。
- 单副本、单 GPU、`Recreate` 策略。一次更新会有短暂停机，不承诺无缝滚动更新。
- Pod 请求 4 CPU / 40GiB，限制 8 CPU / 48GiB；共享内存 2GiB、临时目录上限 4GiB。
- MOSS 内存利用率参数 0.25 是 vLLM 预算，不是实测 GPU 峰值；cgroup 的 memory.peak 也不能当作纯 GPU 显存峰值。
- 离线同时处理一个请求，其余明确 busy；实时可并存，单例并发成功不等于并发 SLA。

## 验证与回滚

部署制品先通过隔离真实 HTTP/WS 检查，再切换现有工作负载。更新后核对实际 Pod 的源码哈希与镜像摘要链，并通过原 HTTPS 网关验证，不只检查 localhost 健康接口。

`docker save` 离线导入可能使 Pod 的 imageID 指向外层 archive index。必须验证该 index **唯一指向本次 release index**，再核对唯一 ARM64 manifest 及 config；不把未知摘要直接放行，也不将外层 index 误报为平台子清单。

当前发布为 `20260922-xfusion-voiceprint-windows`（基于 cleanup 镜像，仅更新部署时尚未提交的 `meeting.py` 与网页，实际源码以镜像内清单哈希为准）。证据、前一版 Deployment、数据备份和带当前 spec/UID 守卫的回滚 patch 位于 `~/.local/share/voiceprint-windows-20260922/`。对应本地证据为 `output/xfusion-voiceprint-windows-20260922/`（不进入 Git/镜像）。本轮回滚恢复前一版 `20260922-xfusion-cleanup`，不还原用户数据。

原迁移数据备份/证据仍保留在 `~/.local/share/voiceprint-deploy-20260921/`，中间 UI 发布位于 `~/.local/share/voiceprint-ui-20260921/`；不要把旧发布的回滚守卫直接用于当前镜像。旧镜像仍在 containerd 中，回滚不自动还原数据备份，以免覆盖更新后用户新增的录音/声纹。

确认目标身份及当前发布仍匹配，并检查没有上传、转写或录音后，可使用本次生成的带 UID/发布守卫的回滚 patch：

```bash
ssh xfusion 'docker exec -i rancher kubectl -n voiceprint-asr-ab-20260920 patch deploy voiceprint-ab --type=json --patch-file=/dev/stdin < "$HOME/.local/share/voiceprint-windows-20260922/rollback.patch.json"'
```

该命令会重建应用 Pod，仅在明确要求回滚时执行。其他应用与 Nginx 不需要重启。

## 2026-09-22 后台任务修复（已上线）

用户确认切换后，复核连接/离线任务/录音 FD 均为空，于 11:50:23～11:50:55（UTC+08）按既有 Recreate 策略更新。新 Pod 从创建到 Ready 为 31 秒，不将其冒称独立测量的端到端停机时间。

镜像 `registry.bsoft.com.cn/ssdev/voiceprint-moss:20260922-xfusion-refresh-r2` 在 xfusion 本地构建并离线导入，未推送仓库；发布 index 为 `sha256:f404e8769e32842d327b2c2b395c4de09421f56efbd37e4e8762c00e0cbca8c5`。实际 Pod `voiceprint-ab-69c9c99dc5-nm64b`，Ready/restart0，archive→release→ARM64 manifest 链及 20 个运行源文件哈希已核验。

- 网页连接、后台转写任务、MOSS 常驻模型分离；刷新/重新打开恢复同一任务，正常显式取消保留模型。
- 本地及 ARM64 候选镜像各 108 项 Python 测试：107 通过、1 条件跳过；Node 12/12。
- 同机隔离真实 MOSS：完整 103.032s 测试录音生成中断连后恢复，42 段全部字段与上一版完全一致，首段约 0.55s，完成约 8.10s；冷启动取消、生成中取消及随后新任务均保持 worker/EngineCore PID 不变。不是医学准确率或总推理提速结论。
- 首个完整实时模型副本的 canary 遇到 CUDA 内存不足并已停止；通过的离线 canary 使用同一候选镜像、相同 MOSS GPU 参数，**仅测试启动时不重复加载实时模型，CAM++ 在 CPU**。它不证明额外实时副本并发的资源容量。测试容器已停止，原声纹及其他服务未变。
- 生产 HTTPS 完整 19.968s 输入：首次冷启动约 40.65s，2 段正常 done。真实 Chrome 经 HTTPS/SSH 上传完整 103.032s 输入，首段显示约 1.38s；生成中刷新并恢复至 42 段，约 8.79s 完成；再次重新打开页面仍恢复同一任务，期间仅一次上传/推理。
- 浏览器显式取消另一个测试任务后，部分结果保持未完成、禁用导出/摘要；随后完整短录音成功。整个浏览器验收前后生产 MOSS worker/EngineCore PID 均保持 106/130，不因刷新或正常取消重载模型。
- 只变更镜像与两个发布/源清单注解；原声纹哈希、切换前两份录音的元数据及六个其他服务启动/重启计数不变。验收期间另有一份新增录音，保留且未读取内容，不以目录新增误判为数据丢失。

证据位于本地 `output/xfusion-refresh-20260922/`、远端 `~/.local/share/voiceprint-refresh-20260922/`。本次未再跑完整 MDT、物理麦克风或重复并发 SLA 验收；不与既有 SSE 发布证据混同。任务恢复受缓存保留期限制，不支持服务进程重启后的推理续跑，详见 [任务接口](api.md#后台任务与刷新恢复)。

## 2026-09-22 声纹与工具边界修复（已上线）

按用户授权于 14:02:13～14:02:44（UTC+08）更新为 `20260922-xfusion-cleanup`，Pod 为 `voiceprint-ab-6985875ccd-nmjl8`，Ready/restart0。Pod 创建至 Ready 为 30 秒，不是独立测量的端到端停机时间。

- 镜像 `registry.bsoft.com.cn/ssdev/voiceprint-moss:20260922-xfusion-cleanup-r1`，发布 index `sha256:b84f8a818b6238eee9175af0eb971ed1b67004cdf39914ec8c2902bd76a982d7`；本地构建、离线导入，未推送 registry。
- 基于前一 refresh 镜像，仅更新 `app/core.py`、`app/server.py`、`app/services/summarizer.py`、`app/utils/voiceprint.py` 和源清单。镜像中本来就没有已删除的旧离线示例；CLI 包装脚本作为测试挂载验证，不额外打进业务镜像。
- 部署配置只修改镜像及两个发布/源清单注解；模型、MOSS 环境、GPU 参数、PVC、资源及安全配置不变。继续使用已部署的 Fun-ASR `3d49f7b` 快照，不将其等同于仓库 gitlink。
- ARM64 镜像内 Python 123 项：122 通过、1 外部服务条件跳过。首次测试仅因新增 shell 夹具位于 noexec 临时目录失败；改为仅测试容器的可执行临时目录后通过，没有修改业务镜像或部署的安全配置。
- 隔离 CPU 实例完成真实 CAM++ 注册/删除和双音频比对，覆盖空文件 422、超限 413、正常注册 200；测试上传上限临时设为 1,000,000 字节，生产仍使用默认 256 MiB。完整 19.968 秒实时输入得到 9 个最终片段，无 error/degraded，保存 PCM 逐字节一致；仅删除隔离实例自己的测试记录。不是物理麦克风或身份准确率验收，也未创建第二套 GPU/MOSS 模型。
- 生产 HTTPS 验证新注册空文件返回 422；完整同一短录音 MOSS 首段约 39.85 秒、完成约 40.08 秒，包含首次冷加载，2 段正常 done，终稿字段与上一版一致。未调用摘要服务；未重跑完整 MDT 或浏览器验收，未变的 MOSS/前端证据与本轮回归分开保留。
- 切换前再次检查连接/任务/录音均空闲并备份数据。声纹文件哈希和原有 3 份录音的大小/mtime_ns 未变，六个其他服务启动时间/重启次数未变，Nginx 未重载。已核验实际 archive→release→ARM64 镜像链及 20 个运行源文件哈希。隔离测试容器已停止。

## 2026-09-22 段内声纹多窗口验证（已上线）

按用户授权于 15:51:01～15:51:33（UTC+08）更新为 `20260922-xfusion-voiceprint-windows`，Pod `voiceprint-ab-7b5d5d6644-cmw8n` Ready/restart0。此时间是切换至就绪窗口，不是独立测量的端到端停机时间。

- 镜像 `registry.bsoft.com.cn/ssdev/voiceprint-moss:20260922-xfusion-voiceprint-windows-r1`，发布 index `sha256:331f7dec1573e424ed69abe80472ddc27e4d07762ce395978066caf75cc5c173`。本地构建、离线导入，未推送 registry；部署时尚未提交 Git，运行源码以 20 文件清单为准。只更新应用两文件、清单及发布注解，不变更依赖、模型或资源配置。
- 长段补充中部/尾部窗口，每窗仍执行原阈值与候选保护；通过身份冲突或存在强候选外赢家时拒识，不跨片段继承姓名。网页区分拒识、有效音频不足和未选参会人，不再将其统称为 0 分。字段与取样规则见 [API](api.md#json)。
- 本地及 ARM64 候选镜像 Python 各 131 项：130 通过、1 外部服务条件跳过；网页 Node 13/13。隔离 CPU CAM++ 使用部署声纹库只读挂载及本次 146.264s 输入，MOSS 片段复用旧版本实际结果；目标段命中，既有确认身份、短段结果未变，单选错误参会人仍全部拒识。该候选验证不冒充第二套真实 MOSS 推理。
- 切换后经原 HTTPS 网关重新处理完整音频，19 段正常 done，文字、时间和匿名标签与旧版逐项一致；00:34 段命中许根桃，实际分数 0.3033（显示 0.3），其余原已确认身份及短段结果未变。首段 39.35s、完成 45.11s，含 worker 冷加载，不与旧版预热 6.37s 直接比较性能。
- Chrome 先经临时 SSH 转发回放同一真实任务并刷新恢复，随后关闭本次转发，直接访问当前 LAN 地址 `https://10.16.78.160:8443/voiceprint/client` 复核 19 段、目标身份及拒识分数展示；没有重新上传或调用摘要。声纹哈希与原有 4 份录音元数据未变，六个其他服务启动/重启状态未变。候选测试容器已退出。
- 该段仍接近阈值，本次单音频回归不代表声纹库质量、广泛误识率或医学准确率验收。未修改已有结果缓存，旧任务需重新处理才使用新规则。

## 尚未由部署验收证明的事项

- 医学数字、术语、否定词与身份归属仍需人工审核；模型成功完成不等于临床准确率验收。
- 浏览器模拟麦克风验证不等于用户物理麦克风/设备兼容性验证。
- 现场旧部署未配置 LLM 地址/密钥，本次保持这一状态，不擅自复制本机 `.env` 或调用外部摘要服务。启用摘要须另行配置获准的服务与数据授权。
- 沿用现场自签名证书及 LAN 访问规则；本次不是新增认证、证书信任体系或正式医疗合规审计。
