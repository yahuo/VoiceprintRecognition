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
    "speaker_threshold": 0.30,      # 声纹匹配阈值
    "min_confidence": 0.15,         # 低置信度过滤（低于此值丢弃）
    "silence_duration": 0.5,        # 静音切分阈值（秒）
    "inheritance_timeout": 3.0,     # 说话人继承超时（秒）
    "silence_energy": 500,          # 静音能量阈值
    # LLM 会议总结配置 (兼容 OpenAI / DeepSeek / GLM / Kimi 等所有 OpenAI 兼容接口)
    "llm_base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    "llm_api_key": os.environ.get("LLM_API_KEY", ""),
    "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    "vad_model_path": os.environ.get("VAD_MODEL_PATH", ""),
    "asr_model_path": os.environ.get("ASR_MODEL_PATH", ""),
    "spk_model_path": os.environ.get("SPK_MODEL_PATH", ""),
}


# ========== 声纹数据库路径 ==========

VOICEPRINT_DB_DIR = os.path.join(PROJECT_ROOT, "voiceprint_db")
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

def merge_diarization_segments(segments: list, gap_threshold_ms: int = 500) -> list:
    """
    合并同一说话人的相邻短片段，减少推理次数

    Args:
        segments: [(start_ms, end_ms, speaker_id), ...]
        gap_threshold_ms: 同一说话人相邻片段间隔小于此值时合并

    Returns:
        合并后的片段列表
    """
    if not segments:
        return segments

    merged = [segments[0]]
    for start_ms, end_ms, speaker in segments[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        if speaker == prev_speaker and (start_ms - prev_end) < gap_threshold_ms:
            merged[-1] = (prev_start, end_ms, speaker)
        else:
            merged.append((start_ms, end_ms, speaker))

    if len(merged) < len(segments):
        print(f"片段合并: {len(segments)} -> {len(merged)} (减少 {len(segments) - len(merged)} 个碎片段)")

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
        self.spk_model = None
        self.diarization_pipeline = None  # pyannote diarization
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

            self.vad_model = AutoModel(**vad_model_kwargs)
        
        # 2. ASR 模型 (Fun-ASR-Nano)
        print("加载 ASR 模型 (Fun-ASR-Nano)...")
        model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
        fun_asr_dir = os.path.join(PROJECT_ROOT, "Fun-ASR")
        model_py_path = os.path.join(fun_asr_dir, "model.py")

        asr_model_kwargs = {
            "model": model_dir,
            "trust_remote_code": True,
            "remote_code": model_py_path,
            "device": device,
            "disable_update": True,
        }

        asr_model_path = CONFIG.get("asr_model_path")
        if asr_model_path and os.path.exists(asr_model_path):
            asr_model_kwargs["model_path"] = asr_model_path

        self.asr_model = AutoModel(**asr_model_kwargs)
        
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

        self.spk_model = AutoModel(**spk_model_kwargs)
        
        # 4. 加载已注册的声纹
        self.reload_voiceprints()
        
        self.is_loaded = True
        print(f"✅ 模型加载完成！")
    
    def reload_voiceprints(self):
        """重新加载声纹库"""
        self.registered_embeddings = load_voiceprint_embeddings()
        print(f"已加载 {len(self.registered_embeddings)} 个注册声纹")
    
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
                        print("提示: 可运行 `python scripts/download_pyannote.py` 下载离线模型")
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
    
    def diarize(self, audio_path: str, audio_data: np.ndarray = None) -> list:
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
            print("正在进行说话人分离...")

            # 预处理音频：统一转换为 16kHz WAV 格式，避免采样率不匹配问题
            import librosa
            import soundfile as sf
            import tempfile

            if audio_data is not None:
                audio = audio_data
            else:
                audio, sr = librosa.load(audio_path, sr=16000)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, 16000)
                processed_audio_path = tmp.name
            
            try:
                output = self.diarization_pipeline(processed_audio_path)
                
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
                # 清理临时文件
                if os.path.exists(processed_audio_path):
                    os.unlink(processed_audio_path)
            
        except Exception as e:
            print(f"说话人分离失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """从音频文件提取声纹"""
        try:
            res = self.spk_model.generate(input=audio_path)
            if res and len(res) > 0:
                emb = res[0].get("spk_embedding", None)
                if emb is not None:
                    # 处理 MPS/CUDA tensor: 先转移到 CPU 再转 numpy
                    import torch
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy()
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
