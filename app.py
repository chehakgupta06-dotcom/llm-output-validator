import streamlit as st
import json
from validator import LLMOutputValidator

st.set_page_config(
    page_title="LLM Output Validator",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 LLM Output Validator")
st.markdown("*Structured output extraction with automatic retry logic*")

# ── Sidebar config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    max_retries = st.slider("Max Retries", min_value=1, max_value=5, value=3)
    st.markdown("---")
    st.markdown("**What this validates:**")
    st.markdown("- `product_name` (string)")
    st.markdown("- `rating` (1.0 – 5.0)")
    st.markdown("- `sentiment` (pos/neg/neutral)")
    st.markdown("- `pros` (list of strings)")
    st.markdown("- `cons` (list of strings)")
    st.markdown("- `summary` (string)")

# ── Main input ────────────────────────────────────────────────────────────────
st.subheader("📝 Paste a Product Review")

default_review = """I bought the Sony WH-1000XM5 headphones last month. 
The noise cancellation is absolutely incredible. Sound quality is rich and detailed. 
Battery lasts about 28 hours. However, the ear cups get uncomfortable after 2-3 hours, 
and at ₹29,000 they're quite expensive. Touch controls are sometimes finicky too."""

review_input = st.text_area(
    "Product Review",
    value=default_review,
    height=150,
    placeholder="Paste any product review here..."
)

if st.button("🚀 Validate & Extract", type="primary", use_container_width=True):
    if not review_input.strip():
        st.error("Please enter a review first.")
    else:
        with st.spinner("Calling LLM and validating output..."):
            validator = LLMOutputValidator(max_retries=max_retries)
            result = validator.validate(review_input)

        # ── Results ───────────────────────────────────────────────────────────
        if result["success"]:
            st.success(f"✅ Validated successfully in {result['attempts']} attempt(s)!")

            data = result["data"]
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Product", data["product_name"])
            with col2:
                st.metric("Rating", f"⭐ {data['rating']}/5.0")
            with col3:
                sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                emoji = sentiment_emoji.get(data["sentiment"], "🤔")
                st.metric("Sentiment", f"{emoji} {data['sentiment'].capitalize()}")

            st.markdown(f"**📋 Summary:** {data['summary']}")

            col4, col5 = st.columns(2)
            with col4:
                st.markdown("**✅ Pros**")
                for pro in data["pros"]:
                    st.markdown(f"- {pro}")
            with col5:
                st.markdown("**❌ Cons**")
                for con in data["cons"]:
                    st.markdown(f"- {con}")

            with st.expander("🔧 Raw JSON Output"):
                st.json(data)

        else:
            st.error(f"❌ Failed after {result['attempts']} attempts")
            st.markdown(f"**Last error:** `{result['error']}`")

        # ── Retry log ─────────────────────────────────────────────────────────
        if result.get("raw_outputs"):
            with st.expander("📋 Retry Log"):
                for log in result["raw_outputs"]:
                    st.markdown(f"- {log}")