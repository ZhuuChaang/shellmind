from typing import TypedDict, List, Literal, Optional, Dict, Any
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage


# ---------- Analyzer ----------

class AnalysisState(TypedDict, total=False):
    task_type: Literal["summarization", "qa"]
    input_type: Literal["raw_text", "conversation"]
    input_length: Literal["short", "medium", "long"]
    language: Literal["zh", "en"]


# ---------- Planner ----------

class PlannerPolicy(TypedDict, total=False):
    compression_ratio: float
    output_style: Literal["paragraph", "bullet"]
    tone: Literal["academic", "neutral"]


class PlanState(TypedDict, total=False):
    chain: Literal["summarization_chain", "qa_chain"]
    execution_mode: Literal["single_step", "multi_step"]
    policy: PlannerPolicy
    use_retrieval: bool
    use_citation: bool


# ---------- Execution / Result ----------

class ExecutionState(TypedDict, total=False):
    prompt_template: str
    model_name: str
    status: Literal["pending", "running", "done"]


class ResultState(TypedDict, total=False):
    raw_output: Optional[str]
    final_output: Optional[str]


# ---------- Agent State ----------

class AgentState(MessagesState):
    analysis: AnalysisState
    plan: PlanState
    execution: ExecutionState
    result: ResultState
