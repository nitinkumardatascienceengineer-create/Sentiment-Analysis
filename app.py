import json
import re
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn

# -------------------- Page setup (must be the first Streamlit call) --------------------
st.set_page_config(
    page_title="Movie Review Analyzer",
    page_icon="🎬",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent

# -------------------- Styling --------------------
st.markdown(
    """
    <style>
    .hero {
        text-align: center;
        padding: 1.6rem 1rem 1rem 1rem;
    }
    .hero h1 { margin: 0.4rem 0 0.2rem 0; }
    .hero p { opacity: 0.8; margin: 0; }
    .eyebrow {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(120, 120, 255, 0.15);
        color: #6c6cff;
    }
    .panel {
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.06);
    }
    .small-note { font-size: 0.85rem; opacity: 0.7; }
    .result {
        text-align: center;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 14px;
        font-size: 2.4rem;
    }
    .result.positive { background: rgba(46, 160, 67, 0.15); border: 1px solid #2ea043; }
    .result.negative { background: rgba(218, 54, 51, 0.15); border: 1px solid #da3633; }
    .result-label { font-size: 1.4rem; font-weight: 700; }
    .result-detail { font-size: 0.95rem; opacity: 0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- Model --------------------
class SentimentModel(nn.Module):
    """Embedding -> masked mean pooling -> dropout -> linear (1 logit)."""

    def __init__(self, vocab_size, embedding_dim=64, padding_id=0, dropout=0.3):
        super().__init__()
        self.padding_id = padding_id
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_id
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, input_ids):
        mask = (input_ids != self.padding_id).unsqueeze(-1).float()
        embedded = self.embedding(input_ids)
        summed = (embedded * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        pooled = self.dropout(pooled)
        return self.fc(pooled).squeeze(-1)


# -------------------- Text processing --------------------
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def tokenize(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)  # IMDb reviews contain <br /> tags
    return TOKEN_PATTERN.findall(text)


def encode_text(text, vocab, max_length, padding_id=0, unknown_id=1):
    tokens = tokenize(text)[:max_length]
    ids = [vocab.get(token, unknown_id) for token in tokens]
    ids += [padding_id] * (max_length - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def load_state_dict_safely(model, state_dict, vocab_size, embedding_dim):
    """Load weights; if layer names differ from this file, match them by shape."""
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        remapped = {}
        for name, tensor in state_dict.items():
            if tensor.dim() == 2 and tuple(tensor.shape) == (vocab_size, embedding_dim):
                remapped["embedding.weight"] = tensor
            elif tensor.dim() == 2 and tuple(tensor.shape) == (1, embedding_dim):
                remapped["fc.weight"] = tensor
            elif tensor.dim() == 1 and tuple(tensor.shape) == (1,):
                remapped["fc.bias"] = tensor
        model.load_state_dict(remapped)


@st.cache_resource(show_spinner="Loading model...")
def load_sentiment_model():
    model_path = BASE_DIR / "sentiment_model.pth"
    vocab_path = BASE_DIR / "vocabulary.json"
    config_path = BASE_DIR / "config.json"

    missing = [p.name for p in (model_path, vocab_path, config_path) if not p.exists()]
    if missing:
        return None, None, None, f"Missing file(s): {', '.join(missing)}"

    try:
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        embedding_dim = int(config["embedding_dim"])
        model = SentimentModel(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            padding_id=int(config.get("padding_id", 0)),
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        load_state_dict_safely(model, state_dict, len(vocab), embedding_dim)
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
try:
    analyze = st.button("Analyze sentiment", width="stretch")
except TypeError:  # older Streamlit versions
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
                """
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
                """
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
