#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
import requests

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2:7b"


def call_ollama(user_content: str, model: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})

    url = f"{OLLAMA_HOST}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}

    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "").strip()
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接 Ollama,请确认已启动 Ollama。")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("错误: Ollama 响应超时。")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 502:
            print("错误: 502 Bad Gateway,请检查 Ollama 是否运行、模型是否已拉取(ollama pull)，或显存不足可换小模型(如 qwen2:1.5b)。")
        else:
            print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


def extract_code(text: str, default: str) -> str:
    text = text.strip()
    m = re.search(r"^```(?:\w*)\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1 if lines[0].strip() == "```" else 0
        end = len(lines)
        for i in range(len(lines) - 1, start - 1, -1):
            if lines[i].strip() == "```":
                end = i
                break
        return "\n".join(lines[start:end]).strip()
    return text if text else default


def _lang_hint(ext: str) -> str:
    if ext == ".py":
        return "Python"
    if ext in (".js", ".ts", ".jsx", ".tsx"):
        return "JavaScript/TypeScript"
    if ext in (".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"):
        return "C/C++"
    if ext == ".java":
        return "Java"
    if ext == ".go":
        return "Go"
    return "代码"


def cmd_explain(file: Path, model: str, output: Path | None, no_backup: bool) -> None:
    code = file.read_text(encoding="utf-8", errors="replace")
    if not code.strip():
        print("文件为空，无需解释。")
        return
    lang = _lang_hint(file.suffix)
    prompt = f"""你是一个代码解释助手。请对下面这段{lang}代码添加详细注释，要求：
1. 在关键逻辑、函数、类、复杂语句上方或行尾用注释解释其作用；
2. 注释使用中文，简洁清晰；
3. 只输出「完整的、带注释的源代码」，不要输出任何说明性文字、不要用 markdown 代码块包裹；
4. 保持原有缩进、格式与可运行性，不要修改或删减代码逻辑。"""
    print(f"正在用 {model} 解释代码并生成注释...")
    result = call_ollama(f"{prompt}\n\n```\n{code}\n```", model)
    out_code = extract_code(result, code)
    out_path = output or file
    if out_path.resolve() == file.resolve() and not no_backup:
        backup = file.with_suffix(file.suffix + ".bak")
        backup.write_text(code, encoding="utf-8")
        print(f"已备份原文件到: {backup}")
    out_path.write_text(out_code, encoding="utf-8")
    print(f"已写入: {out_path}")


def cmd_complexity(file: Path, model: str, output: Path | None) -> None:
    code = file.read_text(encoding="utf-8", errors="replace")
    lang = _lang_hint(file.suffix)
    prompt = f"""请分析下面这段{lang}代码的复杂度，用中文简洁输出：
1. 时间复杂度（最好/平均/最坏，大 O 表示）；
2. 空间复杂度；
3. 圈复杂度或逻辑复杂度（若适用）；
4. 简要说明主要耗时/占空间的部分。"""
    print(f"正在用 {model} 分析复杂度...")
    result = call_ollama(f"{prompt}\n\n```\n{code}\n```", model)
    _print_or_save(result, output)


def cmd_review(file: Path, model: str, output: Path | None) -> None:
    code = file.read_text(encoding="utf-8", errors="replace")
    lang = _lang_hint(file.suffix)
    prompt = f"""请对下面这段{lang}代码进行审查，用中文输出：
1. 潜在 bug 或边界问题；
2. 可读性、可维护性改进建议；
3. 性能或安全方面的建议；
4. 是否符合常见最佳实践。"""
    print(f"正在用 {model} 进行代码审查...")
    result = call_ollama(f"{prompt}\n\n```\n{code}\n```", model)
    _print_or_save(result, output)


def cmd_testcase(file: Path, model: str, output: Path | None) -> None:
    code = file.read_text(encoding="utf-8", errors="replace")
    lang = _lang_hint(file.suffix)
    prompt = f"""请为下面这段{lang}代码生成测试用例，要求：

1. **严格依据**文件最前面的注释中的题干/题目要求（输入输出格式、数据范围、题意）来设计用例，不得偏离题干。
2. 每个用例必须包含三项：
   - **输入说明**：简要说明该用例类型（正常/边界/异常）；
   - **具体输入**：按题干要求的格式写出完整输入内容（可直接复制用于运行）；
   - **期望输出（我的输出）**：根据题干和代码逻辑，该输入下程序应有的正确输出，即“我的输出”的期望结果。
3. 至少覆盖：正常情况、边界情况（空输入、极值等）；若题干涉及异常或非法输入，也需给出对应用例。"""
    print(f"正在用 {model} 生成测试用例...")
    result = call_ollama(f"{prompt}\n\n```\n{code}\n```", model)
    _print_or_save(result, output)


def cmd_classify(file: Path, model: str, output: Path | None) -> None:
    code = file.read_text(encoding="utf-8", errors="replace")
    lang = _lang_hint(file.suffix)
    prompt = f"""请对下面这段{lang}代码进行分类与概括，用中文简要输出：
1. 算法类型或设计模式（如：贪心、DP、分治、单例等）；
2. 主要用途或解决的问题；
3. 关键数据结构；
4. 一句话总结。"""
    print(f"正在用 {model} 对代码分类...")
    result = call_ollama(f"{prompt}\n\n```\n{code}\n```", model)
    _print_or_save(result, output)


def cmd_solution(file: Path, model: str, output: Path | None) -> None:
    code = file.read_text(encoding="utf-8", errors="replace")
    lang = _lang_hint(file.suffix)
    prompt = f"""请针对下面这段{lang}代码，给出题解与解法思路说明，用中文输出：
1. 题目或问题描述（若能从代码推断）；
2. 解题思路与算法步骤；
3. 关键代码与逻辑说明；
4. 可选的优化或变体思路。"""
    print(f"正在用 {model} 生成题解说明...")
    result = call_ollama(f"{prompt}\n\n```\n{code}\n```", model)
    _print_or_save(result, output)


def _print_or_save(text: str, output: Path | None) -> None:
    if output:
        output.write_text(text, encoding="utf-8")
        print(f"已写入: {output}")
    else:
        print(text)


COMMANDS = [
    ("explain", "为代码添加解释性注释"),
    ("complexity", "分析代码复杂度"),
    ("review", "代码审查与改进建议"),
    ("testcase", "生成测试用例"),
    ("classify", "代码分类（算法/用途等）"),
    ("solution", "题解与解法思路说明"),
]
# 命令可简写为前 3 字母，list 须完整输入
CMD_ABBR = {name[:3]: name for name, _ in COMMANDS}
ALL_CMD_NAMES = {name for name, _ in COMMANDS} | {"list"}


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("file", type=Path, help="源代码文件路径")
    p.add_argument("-o", "--output", type=Path, default=None, help="输出文件（不指定则打印到终端）")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama 模型，默认 {DEFAULT_MODEL}")


def cmd_list() -> None:
    print("aics 可用命令（除 list 外可简写为前 3 字母，如 exp/com/rev/tes/cla/sol）：")
    for name, help_ in COMMANDS:
        print(f"  aics {name} <文件>  -  {help_}  [可写 aics {name[:3]}]")
    print("  aics list  -  显示所有命令（须完整输入 list）")


def main() -> None:
    # 支持 aics <文件> <命令>：若第一个不是命令、第二个是命令，则交换为 aics <命令> <文件>
    if len(sys.argv) >= 3:
        a, b = sys.argv[1], sys.argv[2]
        a_not_cmd = a.lower() not in CMD_ABBR and a not in ALL_CMD_NAMES
        b_is_cmd = b.lower() in CMD_ABBR or b in ALL_CMD_NAMES
        if a_not_cmd and b_is_cmd:
            sys.argv[1], sys.argv[2] = sys.argv[2], sys.argv[1]
    if len(sys.argv) >= 2 and sys.argv[1].lower() in CMD_ABBR:
        sys.argv[1] = CMD_ABBR[sys.argv[1].lower()]

    parser = argparse.ArgumentParser(
        prog="aics",
        description="AI 代码分析工具。命令可简写为前 3 字母（如 exp/com/rev），list 须完整输入。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="显示所有命令（须完整输入 list）")

    p_explain = sub.add_parser("explain", help="为代码添加解释性注释")
    _add_common_args(p_explain)
    p_explain.add_argument("--no-backup", action="store_true", help="覆盖原文件时不生成 .bak 备份")

    for name, help_ in COMMANDS:
        if name == "explain":
            continue
        p = sub.add_parser(name, help=help_)
        _add_common_args(p)

    args = parser.parse_args()
    if args.command == "list":
        cmd_list()
        return

    file: Path = args.file
    if not file.exists() or not file.is_file():
        print(f"错误: 文件不存在或不是文件: {file}")
        sys.exit(1)

    model = getattr(args, "model", DEFAULT_MODEL)
    output = getattr(args, "output", None)
    no_backup = getattr(args, "no_backup", False)

    if args.command == "explain":
        cmd_explain(file, model, output, no_backup)
    elif args.command == "complexity":
        cmd_complexity(file, model, output)
    elif args.command == "review":
        cmd_review(file, model, output)
    elif args.command == "testcase":
        cmd_testcase(file, model, output)
    elif args.command == "classify":
        cmd_classify(file, model, output)
    else:
        cmd_solution(file, model, output)


if __name__ == "__main__":
    main()
