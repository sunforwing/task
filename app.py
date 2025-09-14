# app.py
import os
import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from fastapi.responses import StreamingResponse


# ---------------------------
# 自定义远程 Embedding 类 (curl 等价)
# ---------------------------
class RemoteQwenEmbeddings(Embeddings):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        # curl 等价请求
        # curl -X POST -H "Content-Type: application/json" -d '{"texts": ["hello"]}' https://your-endpoint
        payload = {"texts": [text]}
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        # 注意这里根据实际返回调整
        return data["embeddings"][0]


# ---------------------------
# 初始化 FastAPI 应用
# ---------------------------
app = FastAPI(
    title="CardioMind Medical Assistant API",
    description="一个基于RAG和多智能体角色的医疗诊断辅助API。"
)


# ---------------------------
# 模型和向量库的全局初始化
# ---------------------------
vectorstore_path = "faiss_index"

# LLM 初始化
llm = ChatOpenAI(
    model="deepseek-v3-0324",
    openai_api_base="https://api.juheai.top/v1",
    openai_api_key="sk-w6Aw20y5ndUBd9FO6vOPkRCAuo1A5gNYdcz0X8HuB11bJ5h7",   # 记得换成你自己的
    temperature=0
)

# Embedding 改为远程 API
embedding_endpoint = "https://gme-qwen2-vl-7b.ai4s.com.cn/embed/text"
embeddings = RemoteQwenEmbeddings(embedding_endpoint)

# FAISS 向量库加载
try:
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("向量库加载成功！")
except Exception as e:
    print(f"向量库加载失败: {e}")
    vectorstore = None


# ---------------------------
# 你的三个智能体函数（不变）
# ---------------------------
def dr_hypothesis(patient_case: str, vectorstore: FAISS, top_k=3):
    if not vectorstore:
        return "向量库未加载，无法执行RAG检索。"
    # docs = vectorstore.similarity_search(patient_case, k=top_k)
    docs_and_scores = vectorstore.similarity_search_with_score(patient_case, k=top_k)
    docs = [d for d, score in docs_and_scores if score < 0.5]
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
        你是Dr.Hypothesis。根据患者病例信息，结合以下检索到的文献和指南内容，生成一个诊断假设列表：
        病例信息：
        {patient_case}

        检索参考：
        {context}

        请列出可能的诊断假设（每个假设简短描述）。
    """
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    return chain.run({})


def dr_challenger(hypotheses: str):
    prompt = f"""
        你是Dr.Challenger。分析以下诊断假设列表，指出可能的诊断错误，并提出合理替代诊断（如有）。
        诊断假设列表：
        {hypotheses}
    """
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    return chain.run({})


def dr_clinical_reasoning(patient_case: str, hypotheses: str, challenged_hypotheses: str):
    system_msg = SystemMessage(content=(
        "你是Dr.Clinical-Reasoning。根据患者完整病例信息、诊断假设以及Dr.Challenger的分析结果，生成最终诊断。"
        "输出必须严格遵循 JSON 结构，若字段信息没有，请使用空字符串。"
    ))
    user_content = {
        "instruction": "请按照以下顺序输出最终诊断：主诊断（仅一个）、次要诊断（可多个）、鉴别诊断，并保留诊断依据。最终严格输出 JSON。",
        "json_template": {
            "患者信息": {"年龄": "", "性别": "", "入院日期": ""},
            "临床表现": {"主诉": "", "现病史": ""},
            "病史信息": {"既往史": "", "个人史": "", "婚育史": "", "家族史": ""},
            "体格检查": "",
            "辅助检查": "",
            "诊断结果": {
                "主要诊断": {"名称": "", "诊断依据": ["依据1", "依据2"]},
                "次要诊断": [{"名称": "次要诊断1", "诊断依据": ["依据1", "依据2"]}],
                "鉴别诊断": ["鉴别诊断1", "鉴别诊断2"]
            },
            "治疗方案": ["方案1", "方案2"]
        },
        "病例信息": patient_case,
        "诊断假设": hypotheses,
        "Challenger分析": challenged_hypotheses
    }
    user_msg = HumanMessage(content=json.dumps(user_content, ensure_ascii=False, indent=2))
    return llm.stream([system_msg, user_msg])


# ---------------------------
# 定义 API 接口
# ---------------------------
class MedicalRecord(BaseModel):
    medical_record: str


@app.post("/cardiomind")
def generate_diagnosis(record: MedicalRecord):
    if not vectorstore:
        return {"error": "向量库未成功加载，请检查文件路径或重新构建。"}

    patient_case = record.medical_record
    hypotheses = dr_hypothesis(patient_case, vectorstore)
    challenged = dr_challenger(hypotheses)
    response_stream = dr_clinical_reasoning(patient_case, hypotheses, challenged)

    def generate_chunks():
        for chunk in response_stream:
            yield chunk.content
    return StreamingResponse(generate_chunks(), media_type="application/json")


@app.get("/")
def read_root():
    return {"message": "CardioMind API is running!"}
