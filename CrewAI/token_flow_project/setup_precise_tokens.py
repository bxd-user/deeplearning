"""
预先装好「精确 token 统计」：下载并缓存 Qwen2 tokenizer，之后 main.py 将直接使用精确统计。
在 token_flow_project 目录下运行: python setup_precise_tokens.py
"""
import sys

def main():
    print("正在检查并安装精确统计依赖…")
    try:
        import transformers
        print(f"  transformers 已安装 (版本 {transformers.__version__})")
    except ImportError:
        print("  未检测到 transformers，正在安装…")
        import subprocess
        r = subprocess.run([sys.executable, "-m", "pip", "install", "transformers>=4.37.0"], capture_output=True, text=True)
        if r.returncode != 0:
            print("  安装失败，请手动执行: pip install transformers>=4.37.0")
            print(r.stderr or r.stdout)
            sys.exit(1)
        print("  安装完成。")

    print("\n正在下载并缓存 Qwen2 tokenizer（与 qwen2:7b 一致）…")
    try:
        from transformers import AutoTokenizer
        from token_counter import QWEN2_MODEL
        tok = AutoTokenizer.from_pretrained(QWEN2_MODEL, trust_remote_code=True)
        print(f"  tokenizer 已缓存: {QWEN2_MODEL}")
    except Exception as e:
        print(f"  加载失败: {e}")
        print("  请检查网络或稍后重试。main.py 将自动回退为 tiktoken 近似统计。")
        sys.exit(1)

    print("\n验证精确统计…")
    from token_counter import count_tokens, is_using_qwen2
    n = count_tokens("你好，世界。")
    if is_using_qwen2():
        print(f"  示例: \"你好，世界。\" -> {n} tokens (Qwen2)")
        print("\n精确统计已就绪，运行 main.py 时将使用 Qwen2 tokenizer。")
    else:
        print("  当前仍在使用 tiktoken 回退，请检查上方是否有报错。")
        sys.exit(1)


if __name__ == "__main__":
    main()
