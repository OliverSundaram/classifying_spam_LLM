import torch
import tiktoken
import streamlit as st

from model import GPT2Model

MODEL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}

MAX_LENGTH = 120
PAD_TOKEN_ID = 50256
WEIGHTS_PATH = "spam_not_spam_classifying_LLM.pth"

# ─────────────────────────────────────────────────────────────


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2Model(MODEL_CONFIG)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # disables dropout so predictions are deterministic

    # If you deploy to a memory-constrained host and the app crashes or fails to
    # start, uncomment the line below to halve memory usage by switching weights
    # to 16-bit floats. Only do this if you hit a real memory problem.
    # model.half()

    tokenizer = tiktoken.get_encoding("gpt2")
    return model, tokenizer, device


def classify(text: str, model, tokenizer, device) -> tuple[str, float]:
    """Tokenize, pad/truncate, run the model, and return (label, confidence)."""
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    # Truncate long inputs, pad short ones -- the model always expects MAX_LENGTH tokens
    ids = ids[:MAX_LENGTH]
    ids = ids + [PAD_TOKEN_ID] * (MAX_LENGTH - len(ids))

    input_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # add batch dim -> [1, MAX_LENGTH]

    with torch.no_grad():
        logits = model(input_tensor)          # shape: [1, MAX_LENGTH, 2]
        last_token_logits = logits[:, -1, :]  # the last position has attended to the whole sequence
        probs = torch.softmax(last_token_logits, dim=-1)
        pred = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, pred].item())

    label = "Spam" if pred == 1 else "Not spam"
    return label, confidence


st.set_page_config(page_title="spam_classifier", page_icon="▌", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
    }

    .stApp {
        background-color: #0b0f14;
        color: #d8dee9;
    }

    .term-header {
        border-bottom: 1px solid #232a33;
        padding-bottom: 14px;
        margin-bottom: 24px;
    }
    .term-header .path {
        color: #5c6773;
        font-size: 0.85rem;
    }
    .term-header .title {
        color: #e8ecef;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 4px 0 8px 0;
    }
    .term-header .title .cursor {
        color: #d9a441;
    }
    .term-header .desc {
        color: #8b96a3;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .term-header .desc code {
        color: #d9a441;
        background: none;
    }

    .stTextArea textarea {
        background-color: #10161d !important;
        color: #d8dee9 !important;
        border: 1px solid #232a33 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }
    .stTextArea label {
        color: #5c6773 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    .stButton button {
        background-color: transparent;
        color: #d9a441;
        border: 1px solid #d9a441;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        padding: 6px 22px;
    }
    .stButton button:hover {
        background-color: #d9a441;
        color: #0b0f14;
        border: 1px solid #d9a441;
    }

    .log-block {
        background-color: #10161d;
        border: 1px solid #232a33;
        border-left: 3px solid #3a4552;
        border-radius: 3px;
        padding: 16px 20px;
        margin-top: 18px;
        font-size: 0.88rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }
    .log-block .dim { color: #5c6773; }
    .log-block .ok { color: #4fb477; }
    .log-block .warn { color: #d9534f; }
    .log-block .bar { color: #d9a441; letter-spacing: 1px; }

    .term-footer {
        color: #4a5561;
        font-size: 0.78rem;
        margin-top: 40px;
        border-top: 1px solid #1a2028;
        padding-top: 14px;
    }
    .term-footer a { color: #6c7a8a; }
    </style>

    <div class="term-header">
        <div class="title">Email Spam Classifying LLM<span class="cursor">_</span></div>
        <div class="desc">
            A GPT-2 architecture (attention, transformer blocks, layer norm) written from
            scratch in PyTorch, loaded with pretrained GPT-2 weights, then fine-tuned as a
            binary classifier. Paste text into <code>input_text</code> below and run it.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model, tokenizer, device = load_model()

text_input = st.text_area("input_text", height=150, placeholder="paste an email or message here...")

run = st.button("run inference  ▸")

if run:
    if not text_input.strip():
        st.markdown('<div class="log-block warn">error: input_text is empty</div>', unsafe_allow_html=True)
    else:
        with st.spinner(""):
            label, confidence = classify(text_input, model, tokenizer, device)

        token_count = len(tokenizer.encode(text_input, allowed_special={"<|endoftext|>"}))
        filled = int(confidence * 20)
        bar = "█" * filled + "░" * (20 - filled)
        verdict_class = "warn" if label == "Spam" else "ok"

        st.markdown(
            f"""
            <div class="log-block">
<span class="dim">[tokenize]</span> {token_count} tokens → padded to {MAX_LENGTH}
<span class="dim">[inference]</span> forward pass complete
<span class="dim">[result]</span>   label=<span class="{verdict_class}">{label.upper()}</span>  confidence={confidence:.1%}
<span class="bar">{bar}</span> {confidence:.1%}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="term-footer">
        built from scratch · trained and documented in
        <a href="https://github.com/OliverSundaram/building-LLMs" target="_blank">building-LLMs</a>
    </div>
    """,
    unsafe_allow_html=True,
)