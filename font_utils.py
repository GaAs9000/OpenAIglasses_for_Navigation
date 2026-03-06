# -*- coding: utf-8 -*-
"""Shared helpers for locating and loading CJK-capable fonts."""

from functools import lru_cache
import os


_DEFAULT_CJK_FONT_CANDIDATES = [
    # Windows
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simfang.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simsunb.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux (common distros)
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/local/share/fonts/NotoSansCJK-Regular.ttc",
    "~/.local/share/fonts/NotoSansCJK-Regular.ttc",
]


def _normalize_candidates(extra_candidates=None):
    env_font = os.getenv("AIGLASS_CJK_FONT")
    candidates = []
    if env_font:
        candidates.append(env_font)
    candidates.extend(_DEFAULT_CJK_FONT_CANDIDATES)
    if extra_candidates:
        candidates.extend(extra_candidates)
    normalized = []
    seen = set()
    for path in candidates:
        if not path:
            continue
        normalized_path = os.path.normpath(os.path.expanduser(path))
        if normalized_path in seen:
            continue
        seen.add(normalized_path)
        normalized.append(normalized_path)
    return normalized


@lru_cache(maxsize=32)
def _find_cjk_font_path_cached(extra_candidates_key):
    for path in extra_candidates_key:
        try:
            if path and os.path.exists(path):
                return path
        except Exception:
            continue
    return None


def find_cjk_font_path(extra_candidates=None):
    """Return the first available CJK font path, or None if not found."""
    return _find_cjk_font_path_cached(tuple(_normalize_candidates(extra_candidates)))


def load_pil_cjk_font(image_font_module, size, extra_candidates=None):
    """Load a CJK-capable PIL font if possible, otherwise PIL default font."""
    if image_font_module is None:
        return None

    font_path = find_cjk_font_path(extra_candidates)
    if font_path:
        try:
            return image_font_module.truetype(font_path, max(1, int(size)))
        except Exception:
            pass

    try:
        return image_font_module.load_default()
    except Exception:
        return None
