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
import numpy as np
from typing import Dict, List, Tuple, Optional

# 添加 Fun-ASR 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Fun-ASR"))

from funasr import AutoModel


# ========== 配置参数 ==========

CONFIG = {
    "asr_language": "zh",           # 强制中文，避免短音频误判为日语
    "speaker_threshold": 0.30,      # 声纹匹配阈值
    "min_confidence": 0.15,         # 低置信度过滤（低于此值丢弃）
    "silence_duration": 0.5,        # 静音切分阈值（秒）
    "inheritance_timeout": 3.0,     # 说话人继承超时（秒）
    "silence_energy": 500,          # 静音能量阈值
}


# ========== 声纹数据库路径 ==========

VOICEPRINT_DB_DIR = os.path.join(os.path.dirname(__file__), "voiceprint_db")
VOICEPRINT_INDEX_FILE = os.path.join(VOICEPRINT_DB_DIR, "index.json")


# ========== 声纹数据库操作 ==========

def ensure_db_dir():
    """确保声纹数据库目录存在"""
    if not os.path.exists(VOICEPRINT_DB_DIR):
        os.makedirs(VOICEPRINT_DB_DIR)


def load_voiceprint_index() -> Dict[str, str]:
    """加载声纹索引（name -> embedding_file）"""
    if os.path.exists(VOICEPRINT_INDEX_FILE):
        with open(VOICEPRINT_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_voiceprint_index(index: Dict[str, str]):
    """保存声纹索引"""
    ensure_db_dir()
    with open(VOICEPRINT_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def load_voiceprint_embeddings() -> Dict[str, np.ndarray]:
    """加载所有已注册的声纹嵌入"""
    index = load_voiceprint_index()
    embeddings = {}
    
    for name, embedding_file in index.items():
        if os.path.exists(embedding_file):
            emb = np.load(embedding_file).flatten()
            embeddings[name] = emb
    
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
                  threshold: float = None) -> Tuple[str, float]:
    """
    匹配说话人
    
    Args:
        embedding: 待匹配的声纹向量
        registered: 已注册的声纹字典 {name: embedding}
        threshold: 匹配阈值，默认使用 CONFIG["speaker_threshold"]
    
    Returns:
        (speaker_name, score) 元组
    """
    if threshold is None:
        threshold = CONFIG["speaker_threshold"]
    
    if not registered:
        return ("未知", 0.0)
    
    best_name = "未知"
    best_score = 0.0
    
    for name, reg_emb in registered.items():
        score = cosine_similarity(embedding, reg_emb)
        if score > best_score:
            best_score = score
            best_name = name
    
    if best_score >= threshold:
        return (best_name, best_score)
    else:
        return ("未知", best_score)


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
        self.spk_model = None
        self.registered_embeddings = {}
        self.is_loaded = False
    
    def load_models(self, device: str = "cpu", load_vad: bool = True):
        """
        加载所有模型
        
        Args:
            device: 运行设备 ("cpu" 或 "cuda:0")
            load_vad: 是否加载 VAD 模型（实时场景可以不加载）
        """
        print("正在初始化模型...")
        
        # 1. VAD 模型
        if load_vad:
            print("加载 VAD 模型...")
            self.vad_model = AutoModel(
                model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                max_single_segment_time=10000,
                max_end_silence_time=400,
                device=device,
                disable_update=True,
            )
        
        # 2. ASR 模型 (Fun-ASR-Nano)
        print("加载 ASR 模型 (Fun-ASR-Nano)...")
        model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
        fun_asr_dir = os.path.join(os.path.dirname(__file__), "Fun-ASR")
        model_py_path = os.path.join(fun_asr_dir, "model.py")
        
        self.asr_model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code=model_py_path,
            device=device,
            disable_update=True,
        )
        
        # 3. 声纹模型 (CAM++)
        print("加载声纹模型...")
        self.spk_model = AutoModel(
            model="iic/speech_campplus_sv_zh-cn_16k-common",
            device=device,
            disable_update=True,
        )
        
        # 4. 加载已注册的声纹
        self.reload_voiceprints()
        
        self.is_loaded = True
        print(f"✅ 模型加载完成！")
    
    def reload_voiceprints(self):
        """重新加载声纹库"""
        self.registered_embeddings = load_voiceprint_embeddings()
        print(f"已加载 {len(self.registered_embeddings)} 个注册声纹")
    
    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """从音频文件提取声纹"""
        try:
            res = self.spk_model.generate(input=audio_path)
            if res and len(res) > 0:
                emb = res[0].get("spk_embedding", None)
                if emb is not None:
                    return np.array(emb).flatten()
        except Exception as e:
            print(f"声纹提取失败: {e}")
        return None
    
    def transcribe_segment(self, audio_path: str, language: str = None) -> str:
        """
        识别单个音频片段
        
        Args:
            audio_path: 音频文件路径
            language: 语言，默认使用 CONFIG["asr_language"]
        
        Returns:
            识别的文本
        """
        if language is None:
            language = CONFIG["asr_language"]
        
        try:
            res = self.asr_model.generate(
                input=[audio_path],
                language=language,
                use_itn=True,
                batch_size=1
            )
            if res and len(res) > 0:
                return res[0].get("text", "")
        except Exception as e:
            print(f"ASR 识别失败: {e}")
        return ""
    
    def vad_segment(self, audio_path: str) -> List[List[int]]:
        """
        对音频进行 VAD 切分
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            [[start_ms, end_ms], ...] 列表
        """
        if self.vad_model is None:
            raise RuntimeError("VAD 模型未加载")
        
        vad_res = self.vad_model.generate(input=audio_path)
        
        if vad_res and len(vad_res) > 0 and 'value' in vad_res[0]:
            return vad_res[0]['value']
        
        return []


# ========== 说话人状态追踪（用于继承逻辑）==========

class SpeakerTracker:
    """说话人状态追踪器，用于实现说话人继承逻辑"""
    
    def __init__(self, timeout: float = None):
        self.last_speaker = "未知"
        self.last_speech_time = 0
        self.timeout = timeout or CONFIG["inheritance_timeout"]
    
    def update(self, speaker: str, score: float, registered_embeddings: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """
        更新说话人状态，必要时执行继承逻辑
        
        Args:
            speaker: 当前识别的说话人
            score: 当前的置信度
            registered_embeddings: 已注册的声纹字典
        
        Returns:
            (final_speaker, final_score) 元组
        """
        current_time = time.time()
        
        # 说话人继承策略：只有识别为"未知"时才考虑继承
        if speaker == "未知" and \
           (current_time - self.last_speech_time < self.timeout) and \
           self.last_speaker != "未知":
            
            speaker = self.last_speaker
            score = 0.99  # 标记为继承
            print(f"🔄 继承说话人: {self.last_speaker} (因间隔短且本句识别为未知)")
        
        # 更新状态
        if speaker != "未知":
            self.last_speaker = speaker
            self.last_speech_time = current_time
        
        return speaker, score
    
    def reset(self):
        """重置状态"""
        self.last_speaker = "未知"
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
