"""
Token 计数工具。对 qwen2 等模型使用近似估计（字符数/2 更贴近中文混合场景）。
"""

def count_tokens_approx(text: str) -> int:
    """
    近似 token 数：英文约 4 字符/token，中文约 1–2 字符/token，取折中。
    若安装 tiktoken 可替换为精确计数。
    """
    if not text:
        return 0
    # 简单启发式：中文字符多时每字约 1.5 token，英文多时约 4 字符/token
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    total = len(text)
    if total == 0:
        return 0
    other = total - chinese_chars
    return int(chinese_chars * 1.5 + other / 4)
