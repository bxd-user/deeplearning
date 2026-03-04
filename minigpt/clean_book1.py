"""
清洗 data/book1.txt：规范化空白、去除不可见字符，并限制总字符数以适合 8G 显存训练。
输出写入 data/book1_cleaned.txt，保留原文不变。
"""
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_FILE = os.path.join(DATA_DIR, "book1.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "book1_cleaned.txt")
# 8G 显存下与 ollama_corpus 一起训练时，单文件建议不超过约 200 万字符
MAX_CHARS = 2_000_000


def clean(text: str, max_chars=None):
    # 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去除不可见/控制字符（保留 \n \t）
    text = "".join(c for c in text if c in "\n\t" or (c.isprintable() and c != "\u2028"))
    lines = []
    prev_blank = False
    total = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if not prev_blank:
                if max_chars is not None and total + 1 > max_chars:
                    break
                lines.append("")
                total += 1
            prev_blank = True
            continue
        prev_blank = False
        if max_chars is not None and total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    # 末尾不多余空行
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def main():
    if not os.path.isfile(INPUT_FILE):
        print(f"[clean_book1] 未找到 {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[clean_book1] 读取 {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    raw_len = len(raw)
    out = clean(raw, max_chars=MAX_CHARS)
    out_len = len(out)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[clean_book1] 原文 {raw_len} 字符 -> 清洗后 {out_len} 字符 -> {OUTPUT_FILE}")
    if out_len < raw_len and MAX_CHARS:
        print(f"[clean_book1] 已截断至约 {MAX_CHARS} 字符以内，适合 8G 显存与 ollama_corpus 一起训练")
    print("接下来可运行: python run.py（run.py 已使用 book1_cleaned.txt）")


if __name__ == "__main__":
    main()
