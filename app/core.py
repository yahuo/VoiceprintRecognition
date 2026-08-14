#!/usr/bin/env python3
"""
声纹识别核心模块 (Core Module)

集中管理：
- 模型加载与推理服务 (ModelService)
- 声纹匹配逻辑 (match_speaker)
- 声纹数据库操作
- 所有优化参数配置
"""

import os
import sys
import json
import time
import tempfile
import secrets
import shutil
from dataclasses import dataclass
import numpy as np
import librosa
import soundfile as sf
from collections.abc import Collection
from typing import Dict, List, Tuple, Optional

# 添加 Fun-ASR 目录到 Python 路径
# 假设项目根目录为 app/../
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Fun-ASR"))

from funasr import AutoModel

# 加载 .env 环境变量 (确保 CONFIG 初始化前生效)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass


# ========== 配置参数 ==========

CONFIG = {
    "asr_language": "zh",           # 强制中文，避免短音频误判为日语
    "asr_backend": os.environ.get("ASR_BACKEND", "paraformer"),  # paraformer / nano
    "upload_asr_batch_size_s": int(os.environ.get("UPLOAD_ASR_BATCH_SIZE_S", "300")),
    "speaker_threshold": 0.30,      # 声纹匹配阈值
    "offline_registered_match_min_duration_ms": int(os.environ.get("OFFLINE_REGISTERED_MATCH_MIN_DURATION_MS", "3000")),
    "offline_registered_match_score_floor": float(os.environ.get("OFFLINE_REGISTERED_MATCH_SCORE_FLOOR", "0.38")),
    "offline_registered_match_min_margin": float(os.environ.get("OFFLINE_REGISTERED_MATCH_MIN_MARGIN", "0.03")),
    "offline_scoped_match_min_duration_ms": int(os.environ.get("OFFLINE_SCOPED_MATCH_MIN_DURATION_MS", "500")),
    "offline_scoped_match_score_floor": float(os.environ.get("OFFLINE_SCOPED_MATCH_SCORE_FLOOR", "0.30")),
    "offline_scoped_match_min_margin": float(os.environ.get("OFFLINE_SCOPED_MATCH_MIN_MARGIN", "0.0")),
    "offline_scoped_single_match_min_duration_ms": int(os.environ.get("OFFLINE_SCOPED_SINGLE_MATCH_MIN_DURATION_MS", "1500")),
    "offline_scoped_single_match_score_floor": float(os.environ.get("OFFLINE_SCOPED_SINGLE_MATCH_SCORE_FLOOR", "0.38")),
    "offline_scoped_outside_margin": float(os.environ.get("OFFLINE_SCOPED_OUTSIDE_MARGIN", "0.03")),
    "offline_scoped_single_inherit_min_confidence": float(os.environ.get("OFFLINE_SCOPED_SINGLE_INHERIT_MIN_CONFIDENCE", "0.55")),
    "offline_scoped_single_inherit_min_overlap_ratio": float(os.environ.get("OFFLINE_SCOPED_SINGLE_INHERIT_MIN_OVERLAP_RATIO", "0.85")),
    "offline_scoped_single_inherit_min_duration_ms": int(os.environ.get("OFFLINE_SCOPED_SINGLE_INHERIT_MIN_DURATION_MS", "1500")),
    "offline_short_match_min_duration_ms": int(os.environ.get("OFFLINE_SHORT_MATCH_MIN_DURATION_MS", "500")),
    "offline_short_match_score_floor": float(os.environ.get("OFFLINE_SHORT_MATCH_SCORE_FLOOR", "0.48")),
    "offline_short_match_min_margin": float(os.environ.get("OFFLINE_SHORT_MATCH_MIN_MARGIN", "0.10")),
    "offline_ultrashort_match_min_duration_ms": int(os.environ.get("OFFLINE_ULTRASHORT_MATCH_MIN_DURATION_MS", "350")),
    "offline_ultrashort_match_max_duration_ms": int(os.environ.get("OFFLINE_ULTRASHORT_MATCH_MAX_DURATION_MS", "700")),
    "offline_ultrashort_match_score_floor": float(os.environ.get("OFFLINE_ULTRASHORT_MATCH_SCORE_FLOOR", "0.29")),
    "offline_ultrashort_match_min_margin": float(os.environ.get("OFFLINE_ULTRASHORT_MATCH_MIN_MARGIN", "0.07")),
    "offline_registered_match_top_k": int(os.environ.get("OFFLINE_REGISTERED_MATCH_TOP_K", "3")),
    "offline_registered_match_min_support": int(os.environ.get("OFFLINE_REGISTERED_MATCH_MIN_SUPPORT", "2")),
    "offline_registered_match_min_share": float(os.environ.get("OFFLINE_REGISTERED_MATCH_MIN_SHARE", "0.60")),
    "offline_sentence_exact_match_max_duration_ms": int(os.environ.get("OFFLINE_SENTENCE_EXACT_MATCH_MAX_DURATION_MS", "1500")),
    "asr_pause_split_gap_ms": int(os.environ.get("ASR_PAUSE_SPLIT_GAP_MS", "400")),
    "min_confidence": 0.15,         # 低置信度过滤（低于此值丢弃）
    "silence_duration": 0.5,        # 静音切分阈值（秒）
    "inheritance_timeout": 3.0,     # 说话人继承超时（秒）
    "silence_energy": 500,          # 静音能量阈值
    "diarization_merge_gap_ms": int(os.environ.get("DIARIZATION_MERGE_GAP_MS", "800")),
    "diarization_short_segment_ms": int(os.environ.get("DIARIZATION_SHORT_SEGMENT_MS", "1500")),
    "diarization_max_merged_ms": int(os.environ.get("DIARIZATION_MAX_MERGED_MS", "12000")),
    # LLM 会议总结配置 (兼容 OpenAI / DeepSeek / GLM / Kimi 等所有 OpenAI 兼容接口)
    "llm_base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    "llm_api_key": os.environ.get("LLM_API_KEY", ""),
    "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    "vad_model_path": os.environ.get("VAD_MODEL_PATH", ""),
    "asr_model_path": os.environ.get("ASR_MODEL_PATH", ""),
    "punc_model_path": os.environ.get("PUNC_MODEL_PATH", ""),
    "spk_model_path": os.environ.get("SPK_MODEL_PATH", ""),
}


# ========== 声纹数据库路径 ==========

VOICEPRINT_DB_DIR = os.path.join(PROJECT_ROOT, "voiceprint_db")
VOICEPRINT_INDEX_FILE = os.path.join(VOICEPRINT_DB_DIR, "index.json")
VOICEPRINT_FILES_DIR = os.path.join(VOICEPRINT_DB_DIR, "files")

UNKNOWN_SPEAKER_ID = None
UNKNOWN_SPEAKER_NAME = "未知"


@dataclass(frozen=True)
class MatchingScope:
    """一次请求内复用的候选说话人矩阵。"""

    names: Tuple[str, ...]
    matrix: np.ndarray
    is_restricted: bool

    @property
    def size(self) -> int:
        return len(self.names)


# ========== 声纹数据库操作 ==========

def ensure_db_dir():
    """确保声纹数据库目录存在"""
    if not os.path.exists(VOICEPRINT_DB_DIR):
        os.makedirs(VOICEPRINT_DB_DIR)
    if not os.path.exists(VOICEPRINT_FILES_DIR):
        os.makedirs(VOICEPRINT_FILES_DIR)


def _is_new_voiceprint_index(index: dict) -> bool:
    return all(
        isinstance(speaker_id, str)
        and isinstance(entry, dict)
        and entry.get("id") == speaker_id
        and isinstance(entry.get("name"), str)
        and isinstance(entry.get("file"), str)
        for speaker_id, entry in index.items()
    )


def _is_legacy_voiceprint_index(index: dict) -> bool:
    return all(
        isinstance(name, str) and isinstance(embedding_file, str)
        for name, embedding_file in index.items()
    )


def _migrate_legacy_voiceprint_index(index: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """将旧的 name -> file 索引一次性迁移成 id -> {id, name, file}。"""
    ensure_db_dir()
    migrated = {}
    legacy_files = []

    for name, embedding_file in index.items():
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"旧声纹文件不存在: {embedding_file}")

        speaker_id = generate_voiceprint_id()
        while speaker_id in migrated:
            speaker_id = generate_voiceprint_id()

        target_file = _new_voiceprint_file_path()
        shutil.copy2(embedding_file, target_file)
        migrated[speaker_id] = {
            "id": speaker_id,
            "name": name,
            "file": target_file,
        }
        legacy_files.append(embedding_file)

    save_voiceprint_index(migrated)

    migrated_files = {os.path.abspath(entry["file"]) for entry in migrated.values()}
    for embedding_file in set(legacy_files):
        if os.path.abspath(embedding_file) not in migrated_files and os.path.exists(embedding_file):
            os.remove(embedding_file)

    print(f"已迁移旧声纹索引: {len(migrated)} 个声纹")
    return migrated


def load_voiceprint_index() -> Dict[str, Dict[str, str]]:
    """加载声纹索引（id -> {id, name, file}），必要时迁移旧格式。"""
    if os.path.exists(VOICEPRINT_INDEX_FILE):
        with open(VOICEPRINT_INDEX_FILE, "r", encoding="utf-8") as f:
            index = json.load(f)
        if not isinstance(index, dict):
            raise ValueError("声纹索引格式错误: 顶层必须是对象")
        if _is_new_voiceprint_index(index):
            return index
        if _is_legacy_voiceprint_index(index):
            return _migrate_legacy_voiceprint_index(index)
        raise ValueError("声纹索引格式错误: 既不是新格式也不是旧格式")
    return {}


def save_voiceprint_index(index: Dict[str, Dict[str, str]]):
    """保存声纹索引"""
    ensure_db_dir()
    with open(VOICEPRINT_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def generate_voiceprint_id() -> str:
    """生成 24 位 ObjectId 风格的随机 id。"""
    return secrets.token_hex(12)


def _new_voiceprint_file_path() -> str:
    ensure_db_dir()
    while True:
        filename = f"{secrets.token_hex(16)}.npy"
        path = os.path.join(VOICEPRINT_FILES_DIR, filename)
        if not os.path.exists(path):
            return path


def save_voiceprint_embedding(speaker_id: str, name: str, embedding: np.ndarray) -> Dict[str, str]:
    """保存或覆盖一个声纹条目。"""
    index = load_voiceprint_index()
    existing = index.get(speaker_id) or {}
    embedding_file = existing.get("file") or _new_voiceprint_file_path()
    os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
    np.save(embedding_file, embedding)

    entry = {
        "id": speaker_id,
        "name": name,
        "file": embedding_file,
    }
    index[speaker_id] = entry
    save_voiceprint_index(index)
    return entry


def delete_voiceprint_by_id(speaker_id: str) -> Dict[str, str] | None:
    """按 id 删除单个声纹，返回被删除的条目。"""
    index = load_voiceprint_index()
    entry = index.get(speaker_id)
    if entry is None:
        return None

    embedding_file = entry.get("file")
    if embedding_file and os.path.exists(embedding_file):
        os.remove(embedding_file)

    del index[speaker_id]
    save_voiceprint_index(index)
    return entry

def load_voiceprint_embeddings() -> Dict[str, np.ndarray]:
    """加载所有已注册的声纹嵌入"""
    index = load_voiceprint_index()
    embeddings = {}
    
    for speaker_id, entry in index.items():
        embedding_file = entry.get("file")
        if embedding_file and os.path.exists(embedding_file):
            emb = np.load(embedding_file).flatten()
            embeddings[speaker_id] = emb
    
    return embeddings


# ========== 声纹匹配与聚类 ==========

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """计算余弦相似度"""
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1_norm, emb2_norm))


def match_speaker(embedding: np.ndarray, 
                  registered: Dict[str, np.ndarray],
                  threshold: float = None) -> Tuple[str | None, float]:
    """
    匹配说话人
    
    Args:
        embedding: 待匹配的声纹向量
        registered: 已注册的声纹字典 {name: embedding}
        threshold: 匹配阈值，默认使用 CONFIG["speaker_threshold"]
    
    Returns:
        (speaker_id, score) 元组
    """
    if threshold is None:
        threshold = CONFIG["speaker_threshold"]
    
    if not registered:
        return (UNKNOWN_SPEAKER_ID, 0.0)
    
    best_id = UNKNOWN_SPEAKER_ID
    best_score = 0.0
    
    for speaker_id, reg_emb in registered.items():
        score = cosine_similarity(embedding, reg_emb)
        if score > best_score:
            best_score = score
            best_id = speaker_id
    
    if best_score >= threshold:
        return (best_id, best_score)
    else:
        return (UNKNOWN_SPEAKER_ID, best_score)


def cluster_embeddings(embeddings: List[np.ndarray], 
                       n_clusters: int = None, 
                       min_clusters: int = 1, 
                       max_clusters: int = 10) -> List[int]:
    """
    对一组声纹嵌入进行聚类 (用于区分陌生人)
    
    使用谱聚类 (Spectral Clustering)
    
    Args:
        embeddings: 声纹向量列表
        n_clusters: 指定聚类数量 (None 表示自动估计)
        min_clusters: 最小聚类数
        max_clusters: 最大聚类数
    
    Returns:
        labels: 每个向量对应的类别标签 [0, 1, 0, 2...]
    """
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cossim
    except ImportError:
        print("警告: 未安装 scikit-learn，无法执行聚类。请运行 pip install scikit-learn")
        return [0] * len(embeddings)

    if not embeddings:
        return []
    
    X = np.array(embeddings)
    n_samples = X.shape[0]
    
    if n_samples < 2:
        return [0] * n_samples
        
    # 如果指定了聚类数，使用层次聚类
    if n_clusters:
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
        return clustering.labels_.tolist()
    
    # ========== 使用 DBSCAN 自适应聚类 ==========
    # DBSCAN 的优势：
    # 1. 不需要预设聚类数
    # 2. 能自动识别噪声点（异常片段）
    # 3. 基于密度，更适合声纹这种"簇内紧密"的数据
    
    # 计算余弦距离矩阵
    similarity_matrix = sklearn_cossim(X)
    distance_matrix = 1 - similarity_matrix
    distance_matrix[distance_matrix < 0] = 0
    
    # DBSCAN 参数:
    # - eps: 邻域半径 (距离阈值)，余弦距离通常在 0~2 范围
    #   0.65 表示相似度 > 0.35 的样本会被归为同一类
    # - min_samples: 形成一个簇的最小样本数，会议中设为 1 允许单句成簇
    eps = 0.50  # 更严格的阈值，相似度需 > 0.5 才合并
    
    print(f"聚类分析: 使用 DBSCAN，eps={eps:.2f} (相似度阈值≈{1-eps:.2f})")
    
    clustering = DBSCAN(
        eps=eps,
        min_samples=1,  # 允许单个样本成簇
        metric='precomputed'
    ).fit(distance_matrix)
    
    labels = clustering.labels_.tolist()
    
    # DBSCAN 会把噪声标记为 -1，我们需要把它们分配到新的类
    max_label = max(labels) if labels else -1
    for i, label in enumerate(labels):
        if label == -1:
            max_label += 1
            labels[i] = max_label
    
    n_cl = len(set(labels))
    print(f"聚类结果: 发现 {n_cl} 位陌生人")
    
    return labels


# ========== 片段合并 ==========

def merge_diarization_segments(
    segments: list,
    gap_threshold_ms: int = 800,
    short_segment_ms: int = 1500,
    max_merged_duration_ms: int = 12000,
) -> list:
    """
    合并同一说话人的相邻短片段，减少推理次数。

    Args:
        segments: [(start_ms, end_ms, speaker_id), ...]
        gap_threshold_ms: 同一说话人相邻片段间隔小于此值时合并
        short_segment_ms: 前后任一片段很短时，优先合并
        max_merged_duration_ms: 合并后单段最大时长，避免过度合并

    Returns:
        合并后的片段列表
    """
    if not segments:
        return segments

    merged = [segments[0]]
    for start_ms, end_ms, speaker in segments[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        gap_ms = start_ms - prev_end
        prev_duration_ms = prev_end - prev_start
        current_duration_ms = end_ms - start_ms
        merged_duration_ms = end_ms - prev_start

        can_merge = (
            speaker == prev_speaker
            and gap_ms <= gap_threshold_ms
            and merged_duration_ms <= max_merged_duration_ms
            and (
                gap_ms <= gap_threshold_ms // 2
                or prev_duration_ms <= short_segment_ms
                or current_duration_ms <= short_segment_ms
            )
        )

        if can_merge:
            merged[-1] = (prev_start, end_ms, speaker)
            continue

        merged.append((start_ms, end_ms, speaker))

    if len(merged) < len(segments):
        print(
            "片段合并: "
            f"{len(segments)} -> {len(merged)} "
            f"(gap<={gap_threshold_ms}ms, short<={short_segment_ms}ms, "
            f"max<={max_merged_duration_ms}ms)"
        )

    return merged


# ========== 工具函数 ==========

def format_time(ms: int) -> str:
    """将毫秒转换为 MM:SS 格式"""
    seconds = ms // 1000
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02d}:{secs:02d}"


# ========== 模型服务 ==========

class ModelService:
    """统一的模型加载和推理服务"""
    
    def __init__(self):
        self.vad_model = None
        self.asr_model = None
        self.asr_backend = (CONFIG["asr_backend"] or "paraformer").strip().lower()
        if self.asr_backend not in {"paraformer", "nano"}:
            raise ValueError(
                f"不支持的 ASR_BACKEND={self.asr_backend!r}，可选值为 paraformer 或 nano"
            )
        self.spk_model = None
        self.diarization_pipeline = None  # pyannote diarization
        self.registered_embeddings = {}
        self.registered_speakers = {}
        self._emb_names = []
        self._emb_name_to_idx = {}
        self._emb_matrix = None
        self.is_loaded = False

    
    def load_models(self, device: str = "cpu", load_vad: bool = True):
        """
        加载所有模型
        
        Args:
            device: 运行设备 ("cpu" 或 "cuda:0")
            load_vad: 是否加载 VAD 模型（实时场景可以不加载）
        """
        print("正在初始化模型...")

        # 0. CUDA 可用性检查：slim 镜像容易出现 torch.cuda 不可用的情况
        if device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                print("=" * 60)
                print("⚠️  警告: 指定了 CUDA 设备但 torch.cuda.is_available() = False!")
                print("   所有推理将回退到 CPU，性能会严重下降。")
                print("   可能原因:")
                print("   1. venv 中的 PyTorch 是 CPU 版本（检查 pip list | grep torch）")
                print("   2. NVIDIA Container Toolkit 未正确安装")
                print("   3. docker-compose 未配置 GPU 资源（deploy.resources）")
                print("=" * 60)
                device = "cpu"
            else:
                print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")

        # 记录实际使用的设备，供后续 diarization 等组件使用
        self.device = device

        # 1. VAD 模型
        if load_vad:
            print("加载 VAD 模型...")
            vad_model_kwargs = {
                "model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                "max_single_segment_time": 10000,
                "max_end_silence_time": 400,
                "device": device,
                "disable_update": True,
            }
            vad_model_path = CONFIG.get("vad_model_path")
            if vad_model_path and os.path.exists(vad_model_path):
                vad_model_kwargs["model_path"] = vad_model_path
                print(f"  VAD 模型路径: {vad_model_path}")
            elif vad_model_path:
                print(f"  ⚠️ VAD 本地路径不存在: {vad_model_path}，将从网络下载")

            self.vad_model = AutoModel(**vad_model_kwargs)
        
        # 2. ASR 模型：实时、上传和离线链路统一复用同一个后端和模型实例
        self._load_asr_model(device=device)
        
        # 3. 声纹模型 (CAM++)
        print("加载声纹模型...")

        spk_model_kwargs = {
            "model": "iic/speech_campplus_sv_zh-cn_16k-common",
            "device": device,
            "disable_update": True,
        }

        spk_model_path = CONFIG.get("spk_model_path")
        if spk_model_path and os.path.exists(spk_model_path):
            spk_model_kwargs["model_path"] = spk_model_path
            print(f"  SPK 模型路径: {spk_model_path}")
        elif spk_model_path:
            print(f"  ⚠️ SPK 本地路径不存在: {spk_model_path}，将从网络下载")

        self.spk_model = AutoModel(**spk_model_kwargs)
        
        # 4. 加载已注册的声纹
        self.reload_voiceprints()

        self.is_loaded = True
        print(f"✅ 模型加载完成！")

        # 5. CUDA warmup: 用短音频跑一次推理，预编译 CUDA kernel
        #    MPS 不需要此步骤，CUDA 首次推理会编译 kernel 导致延迟
        if device.startswith("cuda"):
            self._cuda_warmup()
    
    def _cuda_warmup(self):
        """CUDA warmup: 用 1 秒静音跑一次推理，触发 kernel 编译和 cuDNN autotuning"""
        print("🔥 CUDA warmup: 预编译推理 kernel...")
        t0 = time.time()
        dummy = np.zeros(16000, dtype=np.float32)  # 1 秒 16kHz 静音
        try:
            self.transcribe_segment(dummy)
        except Exception:
            pass
        try:
            self.extract_embedding(dummy)
        except Exception:
            pass
        print(f"🔥 CUDA warmup 完成，耗时 {time.time() - t0:.1f}s")

    def _load_asr_model(self, device: str):
        """按 ASR_BACKEND 加载唯一的 ASR 模型，供全部转写链路复用。"""
        backend = self.asr_backend
        asr_model_path = CONFIG.get("asr_model_path")

        if backend == "nano":
            print("加载 ASR 模型 (Fun-ASR-Nano)...")
            model_py_path = os.path.join(PROJECT_ROOT, "Fun-ASR", "model.py")
            asr_model_kwargs = {
                "model": "FunAudioLLM/Fun-ASR-Nano-2512",
                "trust_remote_code": True,
                "remote_code": model_py_path,
                "device": device,
                "disable_update": True,
            }
        else:
            print("加载 ASR 模型 (Paraformer)...")
            asr_model_kwargs = {
                "model": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                "vad_model": "fsmn-vad",
                "vad_kwargs": {"max_single_segment_time": 30000},
                "punc_model": "ct-punc",
                "device": device,
                "disable_update": True,
            }

            vad_model_path = CONFIG.get("vad_model_path")
            if vad_model_path and os.path.exists(vad_model_path):
                asr_model_kwargs["vad_model"] = vad_model_path
                print(f"  ASR 复用 VAD 模型目录: {vad_model_path}")
            elif vad_model_path:
                print(f"  ⚠️ ASR 的 VAD 本地路径不存在: {vad_model_path}，将使用默认下载源")

            punc_model_path = CONFIG.get("punc_model_path")
            if punc_model_path and os.path.exists(punc_model_path):
                asr_model_kwargs["punc_model"] = punc_model_path
                print(f"  ASR 复用 PUNC 模型目录: {punc_model_path}")
            elif punc_model_path:
                print(f"  ⚠️ ASR 的 PUNC 本地路径不存在: {punc_model_path}，将使用默认下载源")

        if asr_model_path and os.path.exists(asr_model_path):
            asr_model_kwargs["model_path"] = asr_model_path
            print(f"  ASR 模型路径: {asr_model_path}")
        elif asr_model_path:
            print(f"  ⚠️ ASR 本地路径不存在: {asr_model_path}，将从网络下载")

        try:
            self.asr_model = AutoModel(**asr_model_kwargs)
        except Exception as exc:
            raise RuntimeError(f"ASR 模型加载失败({backend}): {exc}") from exc
        print(f"✅ ASR 模型加载完成 ({backend})")

    def reload_voiceprints(self):
        """重新加载声纹库，并构建预归一化矩阵用于快速匹配"""
        self.registered_speakers = load_voiceprint_index()
        self.registered_embeddings = load_voiceprint_embeddings()
        # 构建预归一化矩阵 (N, D) 用于向量化匹配
        if self.registered_embeddings:
            self._emb_names = list(self.registered_embeddings.keys())
            self._emb_name_to_idx = {
                speaker_id: idx for idx, speaker_id in enumerate(self._emb_names)
            }
            matrix = np.array([self.registered_embeddings[speaker_id] for speaker_id in self._emb_names])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # 防止除零
            self._emb_matrix = matrix / norms
        else:
            self._emb_names = []
            self._emb_name_to_idx = {}
            self._emb_matrix = None
        print(f"已加载 {len(self.registered_embeddings)} 个注册声纹")

    def get_speaker_name(self, speaker_id: str | None) -> str:
        """将注册声纹 id 映射成展示名。"""
        if speaker_id is None:
            return UNKNOWN_SPEAKER_NAME
        entry = self.registered_speakers.get(speaker_id) or {}
        return entry.get("name") or speaker_id

    def format_speaker(self, speaker_id: str | None, fallback_name: str = UNKNOWN_SPEAKER_NAME) -> dict:
        """生成对外返回的说话人字段。"""
        if speaker_id is None:
            return {"speakerId": None, "speaker": fallback_name}
        return {"speakerId": speaker_id, "speaker": self.get_speaker_name(speaker_id)}

    def _normalize_audio_array(self, audio_input):
        """将内存音频统一成 float32 numpy 数组，路径输入保持原样。"""
        if isinstance(audio_input, str):
            return audio_input

        if isinstance(audio_input, bytes):
            audio_array = np.frombuffer(audio_input, dtype=np.int16)
            return audio_array.astype(np.float32) / 32768.0

        if isinstance(audio_input, np.ndarray):
            if np.issubdtype(audio_input.dtype, np.integer):
                return audio_input.astype(np.float32) / 32768.0
            return audio_input.astype(np.float32, copy=False)

        return audio_input

    def _prepare_asr_input(self, audio_input):
        """Fun-ASR Nano 的 remote code 只接受路径或 torch.Tensor。"""
        prepared = self._normalize_audio_array(audio_input)
        if isinstance(prepared, str):
            return prepared

        if isinstance(prepared, np.ndarray):
            import torch
            return torch.from_numpy(np.ascontiguousarray(prepared))

        return prepared

    def _prepare_embedding_input(self, audio_input):
        """声纹模型可直接接受路径或 numpy 音频数组。"""
        prepared = self._normalize_audio_array(audio_input)
        if isinstance(prepared, np.ndarray):
            return np.ascontiguousarray(prepared)
        return prepared

    def _prepare_paraformer_input(self, audio_input):
        """Paraformer 输入预处理。"""
        prepared = self._normalize_audio_array(audio_input)
        if isinstance(prepared, np.ndarray):
            return np.ascontiguousarray(prepared)
        return prepared

    def build_matching_scope(
        self,
        allowed_speaker_ids: Collection[str] | None = None,
    ) -> MatchingScope | None:
        """
        为一次请求预构建候选声纹矩阵，避免每个片段重复筛选。
        """
        if self._emb_matrix is None or len(self._emb_names) == 0:
            return None

        if allowed_speaker_ids is None:
            return MatchingScope(
                names=tuple(self._emb_names),
                matrix=self._emb_matrix,
                is_restricted=False,
            )

        allowed_set = {speaker_id for speaker_id in allowed_speaker_ids if speaker_id in self._emb_name_to_idx}
        candidate_names = tuple(speaker_id for speaker_id in self._emb_names if speaker_id in allowed_set)
        if not candidate_names:
            return MatchingScope(
                names=tuple(),
                matrix=np.empty((0, 0), dtype=np.float32),
                is_restricted=True,
            )

        candidate_indices = [self._emb_name_to_idx[name] for name in candidate_names]
        return MatchingScope(
            names=candidate_names,
            matrix=self._emb_matrix[candidate_indices],
            is_restricted=True,
        )

    def _prepare_matching_candidates(
        self,
        embedding: np.ndarray,
        match_scope: MatchingScope | None = None,
        allowed_speaker_ids: Collection[str] | None = None,
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        根据可选白名单准备候选声纹 id 与相似度分数。

        allowed_speaker_ids:
        - None: 使用全部已注册声纹
        - 空集合: 显式表示无可匹配候选
        """
        if self._emb_matrix is None or len(self._emb_names) == 0:
            return [], None

        q = embedding.flatten()
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [], None
        q = q / q_norm

        resolved_scope = match_scope or self.build_matching_scope(allowed_speaker_ids)
        if resolved_scope is None:
            return [], None

        candidate_names = list(resolved_scope.names)
        candidate_matrix = resolved_scope.matrix
        if not candidate_names or candidate_matrix.size == 0:
            return [], np.array([], dtype=np.float32)

        scores = candidate_matrix @ q
        return candidate_names, scores

    def _best_overall_registered_match(self, embedding: np.ndarray) -> Tuple[str | None, float]:
        """返回全库最相似注册声纹 id 与分数。"""
        if self._emb_matrix is None or len(self._emb_names) == 0:
            return (UNKNOWN_SPEAKER_ID, 0.0)

        q = embedding.flatten()
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return (UNKNOWN_SPEAKER_ID, 0.0)

        q = q / q_norm
        scores = self._emb_matrix @ q
        if scores.size == 0:
            return (UNKNOWN_SPEAKER_ID, 0.0)

        best_idx = int(np.argmax(scores))
        return (self._emb_names[best_idx], float(scores[best_idx]))

    def _scoped_match_conflicts_with_outside_winner(
        self,
        embedding: np.ndarray,
        scoped_name: str,
        scoped_score: float,
        threshold: float,
        match_scope: MatchingScope | None,
    ) -> bool:
        """候选集外若存在更强注册人，则当前 scoped 结果应回退未知。"""
        if match_scope is None or not match_scope.is_restricted or match_scope.size == 0:
            return False

        global_id, global_score = self._best_overall_registered_match(embedding)
        if global_id is None or global_id in match_scope.names:
            return False

        outside_floor = max(threshold, CONFIG["offline_registered_match_score_floor"])
        outside_margin = CONFIG["offline_scoped_outside_margin"]
        return global_score >= outside_floor and (global_score - scoped_score) >= outside_margin

    def match_speaker_fast(
        self,
        embedding: np.ndarray,
        threshold: float = None,
        match_scope: MatchingScope | None = None,
        allowed_speaker_ids: Collection[str] | None = None,
    ) -> Tuple[str | None, float]:
        """
        向量化声纹匹配：单次矩阵乘法替代 Python 循环

        Args:
            embedding: 待匹配的声纹向量
            threshold: 匹配阈值，默认使用 CONFIG["speaker_threshold"]
            allowed_speaker_ids: 可选候选声纹 id 白名单

        Returns:
            (speaker_id, score) 元组，与 match_speaker() 返回格式一致
        """
        if threshold is None:
            threshold = CONFIG["speaker_threshold"]

        candidate_names, scores = self._prepare_matching_candidates(
            embedding,
            match_scope=match_scope,
            allowed_speaker_ids=allowed_speaker_ids,
        )
        if scores is None or len(candidate_names) == 0 or len(scores) == 0:
            return (UNKNOWN_SPEAKER_ID, 0.0)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            return (candidate_names[best_idx], best_score)
        return (UNKNOWN_SPEAKER_ID, best_score)

    def match_registered_speaker_guarded(
        self,
        embedding: np.ndarray,
        threshold: float = None,
        duration_ms: int | None = None,
        match_scope: MatchingScope | None = None,
        allowed_speaker_ids: Collection[str] | None = None,
    ) -> Tuple[str | None, float]:
        """
        离线/上传链路更严格的注册人匹配。

        目标：
        - 短片段不直接认成已注册用户
        - 即使超过基础阈值，也要求达到更高分数下限
        - top1 和 top2 太接近时，视为不确定
        """
        if threshold is None:
            threshold = CONFIG["speaker_threshold"]

        resolved_scope = match_scope or self.build_matching_scope(allowed_speaker_ids)
        if resolved_scope is not None and resolved_scope.is_restricted:
            if resolved_scope.size == 1:
                selected_name = resolved_scope.names[0]
                candidate_names, scores = self._prepare_matching_candidates(embedding)
                if scores is None or len(candidate_names) == 0 or len(scores) == 0:
                    return (UNKNOWN_SPEAKER_ID, 0.0)
                try:
                    selected_idx = candidate_names.index(selected_name)
                except ValueError:
                    return (UNKNOWN_SPEAKER_ID, 0.0)
                best_idx = int(np.argmax(scores))
                best_name = candidate_names[best_idx]
                best_score = float(scores[best_idx])
                selected_score = float(scores[selected_idx])
                effective_threshold = max(threshold, CONFIG["offline_scoped_single_match_score_floor"])
                min_duration_ms = CONFIG["offline_scoped_single_match_min_duration_ms"]
                if duration_ms is not None and duration_ms < min_duration_ms:
                    return (UNKNOWN_SPEAKER_ID, selected_score)
                if selected_score < effective_threshold:
                    return (UNKNOWN_SPEAKER_ID, selected_score)
                if (
                    best_name != selected_name
                    and best_score >= max(threshold, CONFIG["offline_registered_match_score_floor"])
                    and (best_score - selected_score) >= CONFIG["offline_scoped_outside_margin"]
                ):
                    return (UNKNOWN_SPEAKER_ID, selected_score)
                if best_name != selected_name:
                    return (UNKNOWN_SPEAKER_ID, selected_score)
                return (selected_name, selected_score)
            effective_threshold = max(threshold, CONFIG["offline_scoped_match_score_floor"])
            min_duration_ms = CONFIG["offline_scoped_match_min_duration_ms"]
            min_margin = CONFIG["offline_scoped_match_min_margin"]
        else:
            effective_threshold = max(threshold, CONFIG["offline_registered_match_score_floor"])
            min_duration_ms = CONFIG["offline_registered_match_min_duration_ms"]
            min_margin = CONFIG["offline_registered_match_min_margin"]

        candidate_names, scores = self._prepare_matching_candidates(
            embedding,
            match_scope=resolved_scope,
            allowed_speaker_ids=allowed_speaker_ids,
        )
        if scores is None or len(candidate_names) == 0 or len(scores) == 0:
            return (UNKNOWN_SPEAKER_ID, 0.0)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        second_best_score = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else -1.0

        if duration_ms is not None and duration_ms < min_duration_ms:
            return (UNKNOWN_SPEAKER_ID, best_score)

        if best_score < effective_threshold:
            return (UNKNOWN_SPEAKER_ID, best_score)

        if len(scores) > 1 and (best_score - second_best_score) < min_margin:
            return (UNKNOWN_SPEAKER_ID, best_score)

        if self._scoped_match_conflicts_with_outside_winner(
            embedding,
            candidate_names[best_idx],
            best_score,
            threshold,
            resolved_scope,
        ):
            return (UNKNOWN_SPEAKER_ID, best_score)

        return (candidate_names[best_idx], best_score)

    def match_registered_speaker_short_window(
        self,
        embedding: np.ndarray,
        threshold: float = None,
        duration_ms: int | None = None,
        match_scope: MatchingScope | None = None,
        allowed_speaker_ids: Collection[str] | None = None,
    ) -> Tuple[str | None, float]:
        """
        对短句使用更保守的 exact-window 注册人匹配。

        只在以下条件下放行：
        - 片段时长达到最低短句阈值，但仍短于常规 guarded 阈值
        - top1 分数显著高于短句下限
        - top1 与 top2 留出更大的 margin
        """
        if threshold is None:
            threshold = CONFIG["speaker_threshold"]

        if self._emb_matrix is None or len(self._emb_names) == 0:
            return (UNKNOWN_SPEAKER_ID, 0.0)

        guarded_min_duration_ms = CONFIG["offline_registered_match_min_duration_ms"]
        ultrashort_min_duration_ms = CONFIG["offline_ultrashort_match_min_duration_ms"]
        ultrashort_max_duration_ms = CONFIG["offline_ultrashort_match_max_duration_ms"]
        short_min_duration_ms = CONFIG["offline_short_match_min_duration_ms"]
        if duration_ms is None or duration_ms < ultrashort_min_duration_ms or duration_ms >= guarded_min_duration_ms:
            return (UNKNOWN_SPEAKER_ID, 0.0)

        resolved_scope = match_scope or self.build_matching_scope(allowed_speaker_ids)
        candidate_names, scores = self._prepare_matching_candidates(
            embedding,
            match_scope=resolved_scope,
            allowed_speaker_ids=allowed_speaker_ids,
        )
        if scores is None or len(candidate_names) == 0 or len(scores) == 0:
            return (UNKNOWN_SPEAKER_ID, 0.0)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        second_best_score = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else -1.0

        if resolved_scope is not None and resolved_scope.is_restricted:
            if resolved_scope.size == 1:
                # 单参会人模式语义更接近“验证这个人是否在说话”。
                # 短句 exact-window 容易把陌生人短句直接吸进唯一候选人，
                # 因此直接关闭这条快路径，交给更稳的 cluster/guarded 逻辑。
                return (UNKNOWN_SPEAKER_ID, best_score)
            effective_threshold = max(threshold, CONFIG["offline_scoped_match_score_floor"])
            min_margin = CONFIG["offline_scoped_match_min_margin"]
        elif duration_ms <= ultrashort_max_duration_ms:
            effective_threshold = max(threshold, CONFIG["offline_ultrashort_match_score_floor"])
            min_margin = CONFIG["offline_ultrashort_match_min_margin"]
        elif duration_ms >= short_min_duration_ms:
            effective_threshold = max(threshold, CONFIG["offline_short_match_score_floor"])
            min_margin = CONFIG["offline_short_match_min_margin"]
        else:
            return (UNKNOWN_SPEAKER_ID, best_score)

        if best_score < effective_threshold:
            return (UNKNOWN_SPEAKER_ID, best_score)

        if len(scores) > 1 and (best_score - second_best_score) < min_margin:
            return (UNKNOWN_SPEAKER_ID, best_score)

        if self._scoped_match_conflicts_with_outside_winner(
            embedding,
            candidate_names[best_idx],
            best_score,
            threshold,
            resolved_scope,
        ):
            return (UNKNOWN_SPEAKER_ID, best_score)

        return (candidate_names[best_idx], best_score)

    def match_registered_speaker_consensus(
        self,
        candidates: List[Tuple[np.ndarray | None, int]],
        threshold: float = None,
        match_scope: MatchingScope | None = None,
        allowed_speaker_ids: Collection[str] | None = None,
    ) -> Tuple[str | None, float]:
        """
        对同一 pyannote speaker 的多个代表片段做保守判定。

        规则：
        - 先对每个候选片段应用 guarded 匹配
        - 多个候选片段时，至少要有足够支持票数
        - 若多个注册人得分接近或票数接近，直接回退为"未知"
        """
        if threshold is None:
            threshold = CONFIG["speaker_threshold"]

        top_k = max(1, CONFIG["offline_registered_match_top_k"])
        min_support = max(1, CONFIG["offline_registered_match_min_support"])
        min_share = max(0.0, min(1.0, CONFIG["offline_registered_match_min_share"]))

        valid_candidates = [(emb, duration_ms) for emb, duration_ms in candidates[:top_k] if emb is not None]
        if not valid_candidates:
            return (UNKNOWN_SPEAKER_ID, 0.0)

        decisions: List[Tuple[str, float, int]] = []
        for emb, duration_ms in valid_candidates:
            name, score = self.match_registered_speaker_guarded(
                emb,
                threshold=threshold,
                duration_ms=duration_ms,
                match_scope=match_scope,
                allowed_speaker_ids=allowed_speaker_ids,
            )
            if name is not None:
                decisions.append((name, score, duration_ms))

        if not decisions:
            return (UNKNOWN_SPEAKER_ID, 0.0)

        if len(valid_candidates) == 1:
            name, score, _ = decisions[0]
            return (name, score)

        stats: Dict[str, Dict[str, float]] = {}
        for name, score, duration_ms in decisions:
            entry = stats.setdefault(
                name,
                {"votes": 0, "score_sum": 0.0, "best_score": 0.0, "duration_sum": 0.0},
            )
            entry["votes"] += 1
            entry["score_sum"] += score
            entry["best_score"] = max(entry["best_score"], score)
            entry["duration_sum"] += duration_ms

        ranked = sorted(
            stats.items(),
            key=lambda item: (
                item[1]["votes"],
                item[1]["score_sum"] / item[1]["votes"],
                item[1]["best_score"],
                item[1]["duration_sum"],
            ),
            reverse=True,
        )
        best_name, best_stats = ranked[0]
        best_votes = int(best_stats["votes"])
        recognized_count = len(decisions)

        if best_votes < min_support:
            return (UNKNOWN_SPEAKER_ID, best_stats["best_score"])

        if recognized_count > 1 and (best_votes / recognized_count) < min_share:
            return (UNKNOWN_SPEAKER_ID, best_stats["best_score"])

        if len(ranked) > 1:
            second_name, second_stats = ranked[1]
            if second_stats["votes"] == best_votes:
                return (UNKNOWN_SPEAKER_ID, max(best_stats["best_score"], second_stats["best_score"]))
            if (
                second_stats["votes"] > 0
                and best_votes == 1
                and second_name != best_name
            ):
                return (UNKNOWN_SPEAKER_ID, best_stats["best_score"])

        return (best_name, best_stats["best_score"])
    
    def load_diarization_model(self, device: str = "cpu"):
        """
        加载 pyannote 说话人分离模型
        
        优先检查本地 models/pyannote/diarization/config.yaml
        否则尝试从 HuggingFace 远程加载（需要 HF_TOKEN）
        """
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # PyTorch 2.6+ 兼容性修复: monkey-patch torch.load 强制 weights_only=False
            import torch
            _original_torch_load = torch.load
            def _patched_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return _original_torch_load(*args, **kwargs)
            torch.load = _patched_torch_load
            
            try:
                from pyannote.audio import Pipeline
                
                # 1. 尝试加载本地模型
                local_config_path = os.path.join(PROJECT_ROOT, "models/pyannote/diarization/config.yaml")
                if os.path.exists(local_config_path):
                    print(f"📦 加载本地 Pyannote 模型: {local_config_path}")
                    self.diarization_pipeline = Pipeline.from_pretrained(local_config_path)
                
                # 2. 回退到 HuggingFace 在线加载
                else:
                    hf_token = os.environ.get("HF_TOKEN")
                    if not hf_token:
                        print("⚠️ 未找到本地模型且未配置 HF_TOKEN，跳过 diarization 模型加载")
                        print("提示: 请确保 models/pyannote/diarization 已就绪，或配置 HF_TOKEN 在线加载")
                        return False
                    
                    print("加载在线 Pyannote 模型 (pyannote/speaker-diarization-community-1)...")
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-community-1",
                        token=hf_token
                    )
                
                # 3. 将模型移动到指定设备
                if device.startswith("cuda") or device == "mps":
                    torch_device = torch.device(device)
                    self.diarization_pipeline.to(torch_device)
                
                print("✅ Diarization 模型加载完成！")
                return True
                
            finally:
                # 恢复原始的 torch.load
                torch.load = _original_torch_load
            
        except Exception as e:
            print(f"⚠️ Diarization 模型加载失败: {e}")
            return False
    
    def diarize(self, audio_path: Optional[str], audio_data: np.ndarray = None) -> list:
        """
        使用 pyannote 进行说话人分离

        Args:
            audio_path: 音频文件路径
            audio_data: 已加载的 16kHz numpy 音频数据（可选，避免重复 librosa.load）

        Returns:
            分段列表 [(start_ms, end_ms, speaker_id), ...]
        """
        if self.diarization_pipeline is None:
            print("⚠️ Diarization 模型未加载，回退到 VAD 分段")
            return None

        try:
            import torch
            print("正在进行说话人分离...")

            # 优先内存直传，避免临时文件 I/O
            processed_audio_path = None
            if audio_data is not None:
                waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
                pipeline_input = {"waveform": waveform, "sample_rate": 16000}
            else:
                # 兜底：无内存数据时走文件路径（需转为 16kHz WAV）
                audio, _ = librosa.load(audio_path, sr=16000)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio, 16000)
                    processed_audio_path = tmp.name
                pipeline_input = processed_audio_path

            try:
                output = self.diarization_pipeline(pipeline_input)

                # pyannote 4.0 返回 DiarizeOutput 对象，需要访问 .speaker_diarization
                if hasattr(output, 'speaker_diarization'):
                    diarization = output.speaker_diarization
                else:
                    # 兼容旧版本，直接使用输出
                    diarization = output

                segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_ms = int(turn.start * 1000)
                    end_ms = int(turn.end * 1000)
                    segments.append((start_ms, end_ms, speaker))

                # 统计说话人数量
                speakers = set(seg[2] for seg in segments)
                print(f"✅ 说话人分离完成: 检测到 {len(speakers)} 位说话人，{len(segments)} 个片段")

                return segments
            finally:
                # 清理临时文件（仅文件路径模式才有）
                if processed_audio_path and os.path.exists(processed_audio_path):
                    os.unlink(processed_audio_path)
            
        except Exception as e:
            print(f"说话人分离失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def extract_embedding(self, audio_input) -> Optional[np.ndarray]:
        """从音频输入提取声纹，支持文件路径、PCM bytes 或 numpy 音频数组。"""
        try:
            prepared_input = self._prepare_embedding_input(audio_input)
            res = self.spk_model.generate(input=prepared_input)
            if res and len(res) > 0:
                emb = res[0].get("spk_embedding", None)
                if emb is not None:
                    # 处理 MPS/CUDA tensor: 先转移到 CPU 再转 numpy
                    import torch
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy()
                    return emb.flatten()
        except Exception as e:
            print(f"声纹提取失败: {e}")
        return None
    
    def transcribe_segment(self, audio_input, language: str = None) -> str:
        """
        识别单个音频片段
        
        Args:
            audio_input: 音频输入，支持文件路径、PCM bytes 或 numpy 音频数组
            language: 语言，默认使用 CONFIG["asr_language"]
        
        Returns:
            识别的文本
        """
        if language is None:
            language = CONFIG["asr_language"]

        if self.asr_backend == "paraformer":
            return self.transcribe_full_audio(audio_input).get("text", "")
        
        try:
            prepared_input = self._prepare_asr_input(audio_input)
            res = self.asr_model.generate(
                input=prepared_input,
                language=language,
                use_itn=True,
                batch_size=1,
                max_length=200,
            )
            if res and len(res) > 0:
                return res[0].get("text", "")
        except Exception as e:
            print(f"ASR 识别失败: {e}")
        return ""

    def resolve_live_asr_backend(self) -> str:
        """返回全部转写链路统一使用的 ASR 后端。"""
        return self.asr_backend

    def transcribe_live_segment(self, audio_input) -> str:
        """实时 WebSocket 短片段 ASR。"""
        return self.transcribe_segment(audio_input)

    def transcribe_full_audio(self, audio_input, return_timestamps: bool = False) -> dict:
        """
        使用统一 ASR 模型进行整段识别；Nano 不支持时间戳时由上层回退。
        """
        backend = self.asr_backend

        if return_timestamps and backend != "paraformer":
            return {"text": "", "sentences": []}

        model = self.asr_model
        if model is None:
            return {"text": "", "sentences": []}

        try:
            if backend == "paraformer":
                prepared_input = self._prepare_paraformer_input(audio_input)
                if isinstance(prepared_input, str):
                    audio_array, _ = librosa.load(prepared_input, sr=16000)
                    prepared_input = np.ascontiguousarray(audio_array)
                kwargs = {
                    "input": prepared_input,
                    "batch_size_s": CONFIG["upload_asr_batch_size_s"],
                }
                if return_timestamps:
                    kwargs["sentence_timestamp"] = True
                res = model.generate(**kwargs)
            else:
                prepared_input = self._prepare_asr_input(audio_input)
                res = model.generate(
                    input=prepared_input,
                    language=CONFIG["asr_language"],
                    use_itn=True,
                    batch_size=1,
                    max_length=200,
                )
            return self._normalize_asr_result(res)
        except Exception as e:
            print(f"整段 ASR 失败({backend}): {e}")
            return {"text": "", "sentences": []}

    def _normalize_asr_result(self, result) -> dict:
        """兼容不同 ASR 后端的返回结构。"""
        normalized = {"text": "", "sentences": []}
        if not result:
            return normalized

        first = result[0] if isinstance(result, list) else result
        if not isinstance(first, dict):
            return normalized

        normalized["text"] = first.get("text", "") or ""

        sentence_candidates = None
        for key in ("sentence_info", "sentences", "sentence_timestamp", "sentence_timestamps"):
            value = first.get(key)
            if isinstance(value, list) and value:
                sentence_candidates = value
                break

        if sentence_candidates is None and isinstance(result, list):
            if result and all(isinstance(item, dict) and "text" in item for item in result):
                if any(("start" in item and "end" in item) for item in result):
                    sentence_candidates = result

        if not sentence_candidates:
            return normalized

        sentences = []
        for item in sentence_candidates:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "") or item.get("sentence", "") or ""
            start = item.get("start")
            end = item.get("end")
            token_timestamps = item.get("timestamp")
            if start is None or end is None:
                if isinstance(token_timestamps, (list, tuple)) and len(token_timestamps) >= 2:
                    start = token_timestamps[0][0] if isinstance(token_timestamps[0], (list, tuple)) else token_timestamps[0]
                    end = token_timestamps[-1][1] if isinstance(token_timestamps[-1], (list, tuple)) else token_timestamps[-1]
            if start is None or end is None:
                continue
            normalized_item = {
                "text": text,
                "start_ms": int(round(float(start))),
                "end_ms": int(round(float(end))),
            }
            if isinstance(token_timestamps, list) and token_timestamps:
                normalized_item["token_timestamps"] = [
                    [int(round(float(ts[0]))), int(round(float(ts[1])))]
                    for ts in token_timestamps
                    if isinstance(ts, (list, tuple)) and len(ts) >= 2
                ]
            sentences.append(normalized_item)

        normalized["sentences"] = sentences
        return normalized
    
    def vad_segment(self, audio_input) -> List[List[int]]:
        """
        对音频进行 VAD 切分

        Args:
            audio_input: 音频文件路径(str)或 numpy 音频数组

        Returns:
            [[start_ms, end_ms], ...] 列表
        """
        if self.vad_model is None:
            raise RuntimeError("VAD 模型未加载")

        prepared = self._normalize_audio_array(audio_input)
        vad_res = self.vad_model.generate(input=prepared)
        
        if vad_res and len(vad_res) > 0 and 'value' in vad_res[0]:
            return vad_res[0]['value']
        
        return []


# ========== 说话人状态追踪（用于继承逻辑）==========

class SpeakerTracker:
    """说话人状态追踪器，用于实现说话人继承逻辑"""
    
    def __init__(self, timeout: float = None):
        self.last_speaker = UNKNOWN_SPEAKER_ID
        self.last_speech_time = 0
        self.timeout = timeout or CONFIG["inheritance_timeout"]
    
    def update(self, speaker: str | None, score: float, registered_embeddings: Dict[str, np.ndarray]) -> Tuple[str | None, float]:
        """
        更新说话人状态，必要时执行继承逻辑
        
        Args:
            speaker: 当前识别的声纹 id
            score: 当前的置信度
            registered_embeddings: 已注册的声纹字典
        
        Returns:
            (final_speaker_id, final_score) 元组
        """
        current_time = time.time()
        
        # 说话人继承策略：只有识别为未知时才考虑继承
        if speaker is None and \
           (current_time - self.last_speech_time < self.timeout) and \
           self.last_speaker is not None:
            
            speaker = self.last_speaker
            score = 0.99  # 标记为继承
            print(f"🔄 继承说话人: {self.last_speaker} (因间隔短且本句识别为未知)")
        
        # 更新状态
        if speaker is not None:
            self.last_speaker = speaker
            self.last_speech_time = current_time
        
        return speaker, score
    
    def reset(self):
        """重置状态"""
        self.last_speaker = UNKNOWN_SPEAKER_ID
        self.last_speech_time = 0


# ========== 全局单例 ==========

# 可选：提供一个全局的 ModelService 实例
_global_service: Optional[ModelService] = None


def get_service() -> ModelService:
    """获取全局 ModelService 实例"""
    global _global_service
    if _global_service is None:
        _global_service = ModelService()
    return _global_service
