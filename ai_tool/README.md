# aics - AI 代码分析命令行工具

用本机 Ollama 大模型对源代码做：**解释注释**、**复杂度分析**、**代码审查**、**测试用例生成**、**分类**、**题解说明**。一条命令即可，无需联网 API。

---

## 功能一览

除 `list` 外，命令可**简写为前 3 个字母**：`exp`、`com`、`rev`、`tes`、`cla`、`sol`；`list` 须完整输入。支持两种顺序：`aics <命令> <文件>` 或 `aics <文件> <命令>`。

| 命令 | 说明 |
|------|------|
| `aics exp` / `aics explain <文件>` | 为代码添加中文解释性注释（可写回原文件） |
| `aics com` / `aics complexity <文件>` | 分析时间/空间/圈复杂度 |
| `aics rev` / `aics review <文件>` | 代码审查与改进建议 |
| `aics tes` / `aics testcase <文件>` | 生成测试用例 |
| `aics cla` / `aics classify <文件>` | 代码分类（算法/用途等） |
| `aics sol` / `aics solution <文件>` | 题解与解法思路说明 |
| `aics list` | 显示所有可用命令（须完整输入 list） |

---

## 环境要求

- **Python 3.10+**
- **Ollama**（本地运行大模型）：[https://ollama.com](https://ollama.com) 下载安装并保持运行
- 本工具依赖 `requests`，安装时会自动拉取

---

## 快速开始（三步部署）

### 1. 安装并启动 Ollama，拉取模型

```bash
# 安装 Ollama 后，在终端执行（任选一个模型）：
ollama pull qwen2:7b
# 若显存不足（如 8GB 且被占用），可改用小模型：
# ollama pull qwen2:1.5b
```

### 2. 安装 aics（在项目目录下执行一次）

```bash
cd <本仓库的 ai_tool 目录路径>
pip install -e .
```

例如项目在 `E:\vscode\model\ai_tool` 时：

```bash
cd E:\vscode\model\ai_tool
pip install -e .
```

### 3. 使用

在**任意目录**下执行（将 `main.cpp` 换成你的源文件路径）：

```bash
aics list
aics exp main.cpp
aics com main.cpp
```

若系统提示找不到 `aics`，可改用：

```bash
python -m aics explain main.cpp
```

---

## 使用示例

```bash
# 命令可简写为前 3 字母（list 除外），例如 exp/com/rev/tes/cla/sol
aics list

# 为代码加注释（会覆盖原文件并生成 .bak 备份）
aics exp main.cpp

# 注释输出到新文件，不修改原文件
aics exp main.cpp -o main_commented.cpp

# 覆盖原文件且不生成备份
aics exp main.cpp --no-backup

# 分析复杂度、代码审查保存到文件
aics com main.cpp
aics rev main.cpp -o review.txt

# 生成测试用例、分类、题解
aics tes main.cpp
aics cla main.cpp
aics sol main.cpp -o solution.md

# 指定其他模型（如显存不足时用 1.5b）
aics exp main.cpp --model qwen2:1.5b
```

---

## 参数说明

| 参数 | 说明 |
|------|------|
| `-o, --output` | 输出到文件（不指定时，除 explain 外均打印到终端） |
| `--model` | Ollama 模型名，默认 `qwen2:7b` |
| `--no-backup` | 仅 explain：覆盖原文件时不生成 `.bak` 备份 |

---

## 常见问题

- **502 Bad Gateway**：多为 Ollama 未启动、模型未拉取，或显存不足。先执行 `ollama list` 确认模型存在；显存紧张时用 `--model qwen2:1.5b`。
- **找不到 `aics` 命令**：将 Python 的 `Scripts` 目录加入系统 PATH，或使用 `python -m aics <子命令> <文件>`。
- **Ollama 不在本机或端口不同**：修改 `aics.py` 中的 `OLLAMA_HOST`（默认 `http://localhost:11434`）。
