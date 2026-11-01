        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        model = SentimentModel(
            vocab_size=len(vocab),
            embedding_dim=int(config["embedding_dim"]),
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model, vocab, int(config["max_length"]), None
    except Exception as exc:
        return None, None, None, f"Could not load model files: {exc}"


# -------------------- Interface --------------------
st.markdown(
    """
    <div class="hero">
        <span class="eyebrow">Deep Learning · PyTorch</span>
        <h1>🎬 Movie Review Analyzer</h1>
        <p>Understand the sentiment behind a movie review in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, vocab, max_length, load_error = load_sentiment_model()

if load_error:
    st.error(load_error)
    st.markdown(
        """
        <div class="panel">
            <h3>One-time setup</h3>
            <p class="small-note">
            Upload <b>sentiment_model.pth</b>, <b>vocabulary.json</b>, and
            <b>config.json</b> alongside app.py in your GitHub repository.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Enter your movie review")
review = st.text_area(
    "Review",
    height=190,
    placeholder="Example: The acting was brilliant, the story was engaging, and I loved every minute of this movie...",
    label_visibility="collapsed",
    max_chars=10000,
)
st.markdown(
    '<p class="small-note">Tip: Write a complete review for a more useful prediction.</p>',
    unsafe_allow_html=True,
)
analyze = st.button("Analyze sentiment", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if analyze:
    if not review.strip():
        st.warning("Enter a movie review first.")
    else:
        input_ids = encode_text(review, vocab, max_length)

        with torch.inference_mode():
            logit = model(input_ids)
            positive_probability = torch.sigmoid(logit).item()

        predicted_positive = positive_probability >= 0.5
        confidence = (
            positive_probability if predicted_positive
            else 1 - positive_probability
        )

        if predicted_positive:
            st.markdown(
                f"""
                <div class="result positive">
                    <div>😊</div>
                    <div class="result-label">Positive Review</div>
                    <div class="result-detail">The model predicts a positive sentiment.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result negative">
                    <div>😞</div>
                    <div class="result-label">Negative Review</div>
                    <div class="result-detail">The model predicts a negative sentiment.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction confidence", f"{confidence * 100:.1f}%")
        with col2:
            st.metric(
                "Positive probability",
                f"{positive_probability * 100:.1f}%",
            )

        st.progress(max(0.0, min(1.0, positive_probability)))
        st.caption("Probability scale: 0% negative · 100% positive")

st.markdown(
    """
    <div class="small-note" style="text-align:center; margin-top:2rem;">
        Built with PyTorch and Streamlit · Trained on the IMDb movie review dataset
    </div>
    """,
    unsafe_allow_html=True,
)
