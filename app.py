import io
from typing import Optional, Tuple

import streamlit as st
from transformers import pipeline


@st.cache_resource(show_spinner=False)
def load_detector():
    """
    Cache the Hugging Face pipeline so the model只載入一次。
    roberta-base-openai-detector 是輕量級二分類模型，標籤：
    - Fake: 模型判定為 AI 生成
    - Real: 模型判定為人類撰寫
    """
    return pipeline(
        "text-classification",
        model="roberta-base-openai-detector",
        device=-1,  # CPU
    )


def read_uploaded_file(file) -> str:
    """將上傳檔案內容解碼成文字，忽略無法解碼的字元。"""
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def predict(text: str) -> Optional[Tuple[float, float, float]]:
    """
    回傳 (ai_prob, human_prob, max_confidence)
    - ai_prob: Fake 標籤分數
    - human_prob: Real 標籤分數
    - max_confidence: 最高分數，用於低信心提示
    """
    text = (text or "").strip()
    if not text:
        return None

    detector = load_detector()
    outputs = detector(
        text,
        truncation=True,
        max_length=512,
        return_all_scores=True,
    )[0]
    score_map = {o["label"]: float(o["score"]) for o in outputs}
    ai_prob = score_map.get("Fake", 0.0)
    human_prob = score_map.get("Real", 0.0)
    max_confidence = max(score_map.values()) if score_map else 0.0
    return ai_prob, human_prob, max_confidence


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
            ai_prob, human_prob, max_conf = result
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

            if max_conf < 0.6:
                st.warning("模型信心偏低，結果僅供參考。")

st.markdown("---")
st.caption(
    "隱私提示：所有推論僅在本地端執行，不會上傳或儲存您的文本。輸入過短時，模型信心可能較低。"
)
