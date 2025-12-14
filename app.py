import io
import re
from typing import Dict, List, Optional, Tuple

import streamlit as st
from transformers import pipeline


# 支援多個模型並快取，方便 ensemble
@st.cache_resource(show_spinner=False)
def load_detector(model_name: str):
    return pipeline(
        "text-classification",
        model=model_name,
        device=-1,  # CPU
    )


def read_uploaded_file(file) -> str:
    """將上傳檔案內容解碼成文字，忽略無法解碼的字元。"""
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _heuristic_ai_boost(text: str) -> float:
    """
    若文本包含常見 LLM 自我描述語句，對 AI 機率做微幅加權。
    這些片語在真實人類文本中少見，可提升偵測準確度。
    """
    patterns = [
        r"\bas an ai language model\b",
        r"\bi do not have access to real[- ]time data\b",
        r"\bi don't have browsing capabilities\b",
        r"\bhere (is|are) (a|some) (concise|brief)\b",
        r"\bprovide (bullet points|a summary)\b",
        r"\bi cannot provide personal experiences\b",
        r"\bi am an artificial intelligence\b",
        r"\bi'm an ai\b",
        r"\bglad to help\b",
    ]
    text_lower = text.lower()
    return 0.4 if any(re.search(p, text_lower) for p in patterns) else 0.0


def predict(text: str) -> Optional[Tuple[float, float, float, Dict[str, float]]]:
    """
    回傳 (ai_prob, human_prob, max_confidence, breakdown)
    - ai_prob: Fake 標籤分數
    - human_prob: Real 標籤分數
    - max_confidence: 最高分數，用於低信心提示
    - breakdown: 紀錄各模型輸出，便於除錯
    """
    text = (text or "").strip()
    if not text:
        return None

    # 兩個輕量模型做投票：OpenAI detector + ChatGPT detector
    model_names = [
        "roberta-base-openai-detector",  # Fake / Real
        "Hello-SimpleAI/chatgpt-detector-roberta",  # ChatGPT / Human
    ]

    ai_scores: List[Tuple[float, float]] = []  # (ai, weight)
    human_scores: List[Tuple[float, float]] = []
    breakdown: Dict[str, float] = {}

    for name in model_names:
        clf = load_detector(name)
        outputs = clf(
            text,
            truncation=True,
            max_length=512,
            return_all_scores=True,
        )[0]
        score_map = {o["label"].lower(): float(o["score"]) for o in outputs}

        if "fake" in score_map and "real" in score_map:
            ai_scores.append(score_map["fake"])
            human_scores.append(score_map["real"])
            breakdown[f"{name}_ai"] = score_map["fake"]
            breakdown[f"{name}_human"] = score_map["real"]
        elif "chatgpt" in score_map and "human" in score_map:
            ai_scores.append(score_map["chatgpt"])
            human_scores.append(score_map["human"])
            breakdown[f"{name}_ai"] = score_map["chatgpt"]
            breakdown[f"{name}_human"] = score_map["human"]

    if not ai_scores or not human_scores:
        return None

    # 依各模型置信度 (|ai-human|) 加權平均，偏向高置信模型
    ai_prob = 0.0
    human_prob = 0.0
    weight_sum = 0.0
    for ai, human in zip(ai_scores, human_scores):
        ai_val = ai if isinstance(ai, float) else ai
        human_val = human if isinstance(human, float) else human
        weight = max(abs(ai_val - human_val), 0.1)
        ai_prob += ai_val * weight
        human_prob += human_val * weight
        weight_sum += weight
    ai_prob = ai_prob / weight_sum
    human_prob = human_prob / weight_sum

    # 針對明顯 LLM 片語做微幅加權
    ai_prob = min(1.0, ai_prob + _heuristic_ai_boost(text))
    human_prob = max(0.0, 1.0 - ai_prob) if ai_prob + human_prob == 0 else human_prob

    max_confidence = max(ai_prob, human_prob)
    return ai_prob, human_prob, max_confidence, breakdown


st.set_page_config(
    page_title="AI / Human 文章偵測器",
    page_icon="🧭",
    layout="centered",
)

st.title("🧭 AI / Human 文章偵測器")
st.write(
    "輸入一段文本或上傳文字檔，立即估計該段文字為 **AI 生成** 或 **人類撰寫** 的機率。"
)

# 預設樣例
sample_texts = {
    "AI 範例": (
        "Certainly! Here is a concise, well-structured overview of the requested topic. "
        "As an AI language model, I will provide bullet points, a short summary, and a "
        "polite closing statement to ensure clarity and coherence."
    ),
    "人類範例": (
        "昨天加班到十一點，回家路上突然下起了大雨，路邊攤的豆漿還是溫的，"
        "喝完才覺得這週末一定要補個眠。"
    ),
    "學術範例": (
        "The experiment demonstrates that introducing a lightweight regularization term "
        "improves generalization on small datasets without significantly increasing "
        "computational cost."
    ),
}

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


def load_sample(name: str):
    st.session_state.input_text = sample_texts[name]


col_left, col_right = st.columns([3, 1])
with col_left:
    st.text_area(
        "輸入文字",
        key="input_text",
        height=200,
        placeholder="貼上要偵測的內容，或使用右側範例快速測試。",
    )
with col_right:
    st.markdown("範例文本")
    for label in sample_texts.keys():
        st.button(f"載入 {label}", on_click=load_sample, args=(label,))

uploaded_file = st.file_uploader("或上傳純文字檔 (.txt)", type=["txt"])

text_from_file = ""
if uploaded_file is not None:
    text_from_file = read_uploaded_file(uploaded_file)
    if text_from_file:
        st.success("已讀取檔案內容，將優先使用檔案文字進行判定。")
    else:
        st.warning("檔案內容無法解碼，請確認為 UTF-8 純文字。")

st.markdown("---")

if st.button("開始偵測"):
    text = text_from_file or st.session_state.input_text
    if not text.strip():
        st.warning("請先輸入文字或上傳檔案。")
    else:
        with st.spinner("模型推論中，請稍候..."):
            result = predict(text)

        if result is None:
            st.error("未取得有效輸入，請重試。")
        else:
            ai_prob, human_prob, max_conf, breakdown = result
            st.subheader("結果")
            st.write(
                f"AI 生成機率：**{ai_prob * 100:.1f}%** | 人類撰寫機率：**{human_prob * 100:.1f}%**"
            )

            bar_ai, bar_human = st.columns(2)
            with bar_ai:
                st.progress(ai_prob)
                st.caption("AI 生成")
            with bar_human:
                st.progress(human_prob)
                st.caption("人類撰寫")

            label = "AI 生成" if ai_prob >= human_prob else "人類撰寫"
            st.info(f"模型判斷：**{label}**")

            with st.expander("模型細節", expanded=False):
                for k, v in breakdown.items():
                    st.write(f"{k}: {v:.3f}")

            if max_conf < 0.6:
                st.warning("模型信心偏低，結果僅供參考。")

st.markdown("---")
st.caption(
    "隱私提示：所有推論僅在本地端執行，不會上傳或儲存您的文本。輸入過短時，模型信心可能較低。"
)
