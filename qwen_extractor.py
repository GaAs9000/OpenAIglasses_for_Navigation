# qwen_extractor.py
# -*- coding: utf-8 -*-
from typing import Optional, Tuple
import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)
_MISSING_KEY_WARNED = False

# —— 本地优先映射（可随时扩充/改名）——
LOCAL_CN2EN = {
    "红牛": "Red_Bull",
    "ad钙奶": "AD_milk",
    "ad 钙奶": "AD_milk",
    "ad": "AD_milk",
    "钙奶": "AD_milk",
    "矿泉水": "bottle",
    "水瓶": "bottle",
    "可乐": "coke",
    "雪碧": "sprite",
}

def _make_client() -> Optional[OpenAI]:
    # 复用你百炼兼容端点；支持从环境变量读取
    global _MISSING_KEY_WARNED
    base_url = os.getenv("DASHSCOPE_COMPAT_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        if not _MISSING_KEY_WARNED:
            logger.warning("未设置 DASHSCOPE_API_KEY，Qwen 提取器将回退到本地规则")
            _MISSING_KEY_WARNED = True
        return None
    return OpenAI(api_key=api_key, base_url=base_url)

PROMPT_SYS = (
    "You are a label normalizer. Convert the given Chinese object "
    "description into a short, lowercase English YOLO/vision class name "
    "(1~3 words). If multiple are given, return the single most likely one. "
    "Output ONLY the label, no punctuation."
)

def extract_english_label(query_cn: str) -> Tuple[str, str]:
    """
    返回 (label_en, source)；source ∈ {'local', 'qwen', 'fallback'}
    """
    q = (query_cn or "").strip().lower()
    if q in LOCAL_CN2EN:
        return LOCAL_CN2EN[q], "local"

    # 简单规则：去掉前缀修饰词
    for k, v in LOCAL_CN2EN.items():
        if k in q:
            return v, "local"

    # 调用 Qwen Turbo（兼容 Chat Completions）；缺少 key 或请求失败时回退默认标签
    try:
        client = _make_client()
        if client is None:
            return "bottle", "fallback"
        msgs = [
            {"role": "system", "content": PROMPT_SYS},
            {"role": "user",   "content": query_cn.strip()},
        ]
        rsp = client.chat.completions.create(
            model=os.getenv("QWEN_MODEL", "qwen-turbo"),
            messages=msgs,
            stream=False
        )
        label = (rsp.choices[0].message.content or "").strip()
        # 清洗一下
        label = label.replace(".", "").replace(",", "").replace("  ", " ").strip()
        # 兜底：空就回 'bottle'
        return (label or "bottle"), "qwen"
    except Exception as exc:
        logger.warning("Qwen 提取失败，回退默认标签: %s", exc)
        return "bottle", "fallback"
