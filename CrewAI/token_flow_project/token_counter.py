"""
统计文本的 token 数量，支持两种方式：
- 精确：使用 Qwen2 官方 tokenizer（与 Ollama qwen2:7b 一致），需安装 transformers
- 回退：使用 tiktoken cl100k_base（近似，无需额外依赖）
"""
from typing import Optional

# tiktoken 回退
_DEFAULT_ENCODING = "cl100k_base"
_tiktoken_encoder: Optional[object] = None

# Qwen2 精确统计（与 qwen2:7b 同源）
_qwen2_tokenizer: Optional[object] = None
_use_qwen2: Optional[bool] = None  # None=未尝试, True/False=已确定

QWEN2_MODEL = "Qwen/Qwen2-7B"  # 仅加载 tokenizer，与 Ollama qwen2:7b 一致


def _get_tiktoken_encoder():
    """延迟加载 tiktoken 编码器。"""
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        import tiktoken
        _tiktoken_encoder = tiktoken.get_encoding(_DEFAULT_ENCODING)
    return _tiktoken_encoder


def _get_qwen2_tokenizer():
    """延迟加载 Qwen2 tokenizer，失败则返回 None。"""
    global _qwen2_tokenizer, _use_qwen2
    if _use_qwen2 is False:
        return None
    if _qwen2_tokenizer is not None:
        return _qwen2_tokenizer
    try:
        from transformers import AutoTokenizer
        _qwen2_tokenizer = AutoTokenizer.from_pretrained(QWEN2_MODEL, trust_remote_code=True)
        _use_qwen2 = True
        return _qwen2_tokenizer
    except Exception:
        _use_qwen2 = False
        return None


def is_using_qwen2() -> bool:
    """当前是否使用 Qwen2 精确统计（调用一次 count_tokens 后结果才确定）。"""
    if _use_qwen2 is None:
        _get_qwen2_tokenizer()
    return _use_qwen2 is True


def count_tokens(
    text: str,
    encoding_name: str = _DEFAULT_ENCODING,
    *,
    use_qwen2: bool = True,
) -> int:
    """
    统计一段文本的 token 数量。

    :param text: 待统计文本，可为 None 或空字符串
    :param encoding_name: tiktoken 编码名称（仅在使用 tiktoken 回退时生效）
    :param use_qwen2: 为 True 时优先使用 Qwen2 官方 tokenizer（与 qwen2:7b 一致），失败则用 tiktoken
    :return: token 数量
    """
    if not text or not text.strip():
        return 0
    if use_qwen2:
        tok = _get_qwen2_tokenizer()
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=False))
    enc = _get_tiktoken_encoder()
    return len(enc.encode(text))


def count_tokens_and_report(
    label: str,
    text: str,
    encoding_name: str = _DEFAULT_ENCODING,
    use_qwen2: bool = True,
) -> int:
    """
    统计 token 并打印一行报告。

    :param label: 显示标签（如 "输入" / "输出"）
    :param text: 待统计文本
    :param encoding_name: tiktoken 编码名称（回退时用）
    :param use_qwen2: 是否优先使用 Qwen2 精确统计
    :return: token 数量
    """
    n = count_tokens(text, encoding_name, use_qwen2=use_qwen2)
    method = "Qwen2" if is_using_qwen2() else "tiktoken"
    print(f"  [Token] {label}: {n} tokens ({method})")
    return n
