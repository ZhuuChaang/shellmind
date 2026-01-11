# agent/nodes/qa.py
from agent.state import AgentState
from llm.inference import LocalLLM
from rag.rag import retrieve_rag_chunks  # 假设我们把函数保存为 rag_routines.py

def qa_node_factory(llm: LocalLLM, top_k: int = 5):
    def qa_node(state: AgentState) -> AgentState:
        user_query = state["messages"][-1].content

        # 1. 检查是否需要 RAG
        if state.get("plan", {}).get("use_retrieval", False):
            # 调用 RAG 检索 top-k 文本
            retrieved_chunks = retrieve_rag_chunks(
                query=user_query,
                top_k=top_k
            )
            # 2. 整合到 prompt
            prompt = (
                "Refer to the following document content to answer the question:\n"
                f"{chr(10).join(retrieved_chunks)}\n\n"
                f"user question:{user_query}"
            )
        else:
            prompt = user_query

        # 3. 记录执行状态
        state["execution"] = {
            "prompt_template": "summarization",
            "model_name": "local_llm",
            "status": "running",
        }

        # 4. 调用 LLM
        output = llm.generate(prompt, temperature=0.9, repetition_penalty=1.1)

        # 5. 更新状态
        state["execution"]["status"] = "done"
        state["result"] = {
            "raw_output": output,
            "final_output": output,
            "retrieved_docs": retrieved_chunks if state.get("plan", {}).get("use_retrieval", False) else []
        }

        return state

    return qa_node
