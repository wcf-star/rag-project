# -*- coding: utf-8 -*-
"""RAG Challenge 2 —— Streamlit 可视化问答界面。

基于 MinerU 解析 + 向量检索 + 可切换 LLM（本地 Ollama / 云端千问）的中芯国际年报问答。
运行方式：
    streamlit run app_streamlit.py
"""
import streamlit as st
from pathlib import Path
from src.pipeline import Pipeline, max_config

# 数据集根目录（中芯国际财报研报集）
ROOT_PATH = Path("data/stock_data")
pipeline = Pipeline(ROOT_PATH, run_config=max_config)

st.set_page_config(page_title="RAG", layout="wide")

# 顶部标题
st.markdown("""
<div style='background: linear-gradient(90deg, #7b2ff2 0%, #f357a8 100%); padding: 20px 0; border-radius: 12px; text-align: center;'>
    <h2 style='color: white; margin: 0;'>🚀 RAG SYSTEM</h2>
    <div style='color: #fff; font-size: 16px;'>企业文档问答 | MinerU 解析 + 向量检索 + 本地 Ollama / 云端千问</div>
</div>
""", unsafe_allow_html=True)

# 左侧输入区
with st.sidebar:
    st.header("查询设置")
    user_question = st.text_area(
        "输入问题",
        "请简要总结中芯国际2024年主营业务的主要内容。",
        height=100,
    )
    submit_btn = st.button("生成答案", use_container_width=True)

# 右侧主内容区
st.markdown("<h3 style='margin-top: 24px;'>检索结果</h3>", unsafe_allow_html=True)

if submit_btn and user_question.strip():
    with st.spinner("正在生成答案，请稍候..."):
        try:
            answer = pipeline.answer_single_question(user_question, kind="string")

            step_by_step = answer.get("step_by_step_analysis", "-")
            reasoning_summary = answer.get("reasoning_summary", "-")
            relevant_pages = answer.get("relevant_pages", [])
            final_answer = answer.get("final_answer", "-")

            st.markdown("**分步推理：**")
            st.info(step_by_step)
            st.markdown("**推理摘要：**")
            st.success(reasoning_summary)
            st.markdown("**相关页面：**")
            st.write(relevant_pages)
            st.markdown("**最终答案：**")
            st.write(final_answer)
        except Exception as e:
            st.error(f"生成答案时出错: {e}")
else:
    st.info("请在左侧输入问题并点击【生成答案】")
