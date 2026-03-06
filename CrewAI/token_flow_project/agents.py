"""
最小多 Agent 系统：CrewAI + Ollama (qwen2:7b)。
定义两个协作 Agent 与一个顺序执行的 Crew。
"""
from crewai import Agent, Crew, LLM, Process, Task

# 使用 Ollama 本地模型 qwen2:7b（需先执行 ollama pull qwen2:7b）
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "ollama/qwen2:7b"

# 本地 Ollama 推理较慢，延长超时（秒）避免 502/超时
llm = LLM(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.3,
    timeout=300,
)

# Agent 1：研究员，负责收集与总结信息
researcher = Agent(
    role="研究员",
    goal="根据给定主题收集要点并写出简短总结",
    backstory="你是一名严谨的研究员，善于提炼关键信息。",
    llm=llm,
    verbose=True,
)

# Agent 2：评审员，负责对研究员输出做简要评审
reviewer = Agent(
    role="评审员",
    goal="对研究员的总结做简短评审，指出亮点或改进点",
    backstory="你是一名审稿人，注重逻辑与表达清晰。",
    llm=llm,
    verbose=True,
)


def create_crew(prompt: str):
    """
    根据 prompt 创建并返回一个最小 Crew：
    任务1 = 研究员基于 prompt 写总结，任务2 = 评审员评审该总结。
    """
    task_research = Task(
        description=f"请针对以下主题撰写一段简短总结（3–5 句话）：\n\n{prompt}",
        expected_output="一段 3–5 句话的总结文本",
        agent=researcher,
    )

    task_review = Task(
        description="请对上一名成员给出的总结做简短评审：指出一个亮点和一个可改进点。",
        expected_output="一段简短的评审意见，包含一个亮点和一个改进点",
        agent=reviewer,
        context=[task_research],
    )

    crew = Crew(
        agents=[researcher, reviewer],
        tasks=[task_research, task_review],
        process=Process.sequential,
        verbose=True,
    )
    return crew
