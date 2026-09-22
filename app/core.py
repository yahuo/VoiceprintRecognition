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
import secrets
import shutil
import threading
from dataclasses import dataclass
import numpy as np
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
    "streaming_asr_model_path": os.environ.get("STREAMING_ASR_MODEL_PATH", ""),
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
    "min_confidence": 0.15,         # 低分仅拒绝身份，仍保留文字与未知说话人
    "silence_duration": 0.5,        # 静音切分阈值（秒）
    # LLM 会议总结配置 (兼容 OpenAI / DeepSeek / GLM / Kimi 等所有 OpenAI 兼容接口)
    "llm_base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    "llm_api_key": os.environ.get("LLM_API_KEY", ""),
    "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    "vad_model_path": os.environ.get("VAD_MODEL_PATH", ""),
    "asr_model_path": os.environ.get("ASR_MODEL_PATH", ""),
    "spk_model_path": os.environ.get("SPK_MODEL_PATH", ""),
}


# ========== 声纹数据库路径 ==========

VOICEPRINT_DB_DIR = os.environ.get("VOICEPRINT_DB_DIR", os.path.join(PROJECT_ROOT, "voiceprint_db"))
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


# ========== 声纹匹配 ==========

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
        from .services.moss import MossTranscriber
        self.asr_model = None
        self.spk_model = None
        self.moss = MossTranscriber()
        self.streaming_vad_model = None
        self.streaming_asr_model = None
        self._streaming_vad_lock = threading.Lock()
        self._streaming_asr_lock = threading.Lock()
        self._embedding_lock = threading.Lock()
        self._offline_lock = threading.Lock()
        self.registered_embeddings = {}
        self.registered_speakers = {}
        self._emb_names = []
        self._emb_name_to_idx = {}
        self._emb_matrix = None
        self._asr_inference_lock = threading.Lock()
        self.is_loaded = False

    
    def load_models(self, device: str = "cpu", load_vad: bool = True, *, load_live: bool = True):
        """
        加载所有模型
        
        Args:
            device: 运行设备 ("cpu" 或 "cuda:0")
            load_vad: 兼容旧调用；实时链路始终使用 FSMN-VAD
            load_live: 离线 CLI / 声纹管理可不加载实时 ASR；MOSS 独立懒加载
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

        # 记录 CAM++ / 实时 ASR 实际设备；MOSS 不在此环境加载。
        self.device = device

        if load_live:
            print("加载实时模型 (FSMN-VAD + Paraformer Streaming + Nano)...")
            asr_model_kwargs = {
                "model": "FunAudioLLM/Fun-ASR-Nano-2512",
                "trust_remote_code": True,
                "remote_code": os.path.join(PROJECT_ROOT, "Fun-ASR", "model.py"),
                "device": device, "disable_update": True,
            }
            if CONFIG.get("asr_model_path"):
                asr_model_kwargs["model_path"] = CONFIG["asr_model_path"]
            self.asr_model = AutoModel(**asr_model_kwargs)
            self._load_streaming_models(device)
        
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
        if self.asr_model is not None:
            try:
                self.transcribe_segment(dummy)
            except Exception:
                pass
        try:
            self.extract_embedding(dummy)
        except Exception:
            pass
        print(f"🔥 CUDA warmup 完成，耗时 {time.time() - t0:.1f}s")

    def _load_streaming_models(self, device):
        # FSMN 在 CPU 上运行；每次调用显式传入会话缓存，并隔离 AutoModel 的可变 kwargs。
        self.streaming_vad_model = AutoModel(
            model=CONFIG["vad_model_path"] or "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            device="cpu", disable_update=True, disable_pbar=True,
            max_single_segment_time=60000,
        )
        self.streaming_asr_model = AutoModel(
            model=CONFIG["streaming_asr_model_path"] or "paraformer-zh-streaming",
            device=device, disable_update=True, disable_pbar=True,
        )

    def vad_stream(self, audio, cache, *, is_final, silence_ms):
        if self.streaming_vad_model is None:
            raise RuntimeError("streaming VAD model not loaded")
        with self._streaming_vad_lock:
            result = self.streaming_vad_model.generate(
                input=self._normalize_audio_array(audio), cache=cache,
                chunk_size=200, is_final=is_final,
                max_end_silence_time=silence_ms,
            )
        return result[0].get("value", []) if result else []

    def transcribe_stream_chunk(self, audio, cache, *, is_final):
        if self.streaming_asr_model is None:
            raise RuntimeError("streaming ASR model not loaded")
        with self._streaming_asr_lock:
            result = self.streaming_asr_model.generate(
                input=self._normalize_audio_array(audio), cache=cache,
                is_final=is_final, chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4, decoder_chunk_look_back=1,
            )
        return result[0].get("text", "") if result else ""

    def close(self):
        self.moss.close()

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
        对同一匿名说话人的多个代表片段做保守判定。

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
    
    def extract_embedding(self, audio_input) -> Optional[np.ndarray]:
        """从音频输入提取声纹，支持文件路径、PCM bytes 或 numpy 音频数组。"""
        try:
            prepared_input = self._prepare_embedding_input(audio_input)
            with self._embedding_lock:
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
    
    def transcribe_segment(
        self,
        audio_input,
        language: str = None,
        max_length: int = 200,
        hotwords: Collection[str] | None = None,
    ) -> str:
        """
        识别单个音频片段
        
        Args:
            audio_input: 音频输入，支持文件路径、PCM bytes 或 numpy 音频数组
            language: 语言，默认使用 CONFIG["asr_language"]
            max_length: 最大生成 token 数，短片段默认 200
            hotwords: 可选的少量上下文热词
        
        Returns:
            识别的文本
        """
        if language is None:
            language = CONFIG["asr_language"]
        
        try:
            prepared_input = self._prepare_asr_input(audio_input)
            res = self._generate_nano(
                prepared_input,
                language=language,
                max_length=max_length,
                hotwords=hotwords,
            )
            if res and len(res) > 0:
                return res[0].get("text", "")
        except Exception as e:
            print(f"ASR 识别失败: {e}")
        return ""

    def _generate_nano(
        self,
        prepared_input,
        *,
        language: str,
        max_length: int,
        hotwords: Collection[str] | None,
    ):
        # FunASR AutoModel 会原地更新单例 kwargs；串行调用并显式传空列表，
        # 避免 accuracy 的热词在并发或后续 speed 请求中残留。
        with self._asr_inference_lock:
            return self.asr_model.generate(
                input=prepared_input,
                language=language,
                itn=True,
                hotwords=list(hotwords or ()),
                batch_size=1,
                max_length=max_length,
            )

# ========== 全局单例 ==========

# 可选：提供一个全局的 ModelService 实例
_global_service: Optional[ModelService] = None


def get_service() -> ModelService:
    """获取全局 ModelService 实例"""
    global _global_service
    if _global_service is None:
        _global_service = ModelService()
    return _global_service
