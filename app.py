import streamlit as st
import pandas as pd
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"] {
    background: #080d18;
    color: #e5eaf5;
}
[data-testid="stHeader"] { background: rgba(8,13,24,.96); }
[data-testid="stSidebar"] {
    background: #101827;
    border-right: 1px solid #25324a;
}
[data-testid="stSidebar"] * { color: #dbe5f7; }
.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3, p, label, .stMarkdown { color: #e5eaf5; }
.hero {
    background: linear-gradient(115deg,#101a35 0%,#142c5c 60%,#25206b 100%);
    border: 1px solid #3156b8; padding: 30px 34px; border-radius: 20px;
    color: white; margin-bottom: 22px;
    box-shadow: 0 12px 35px rgba(22,70,180,.16);
}
.hero h1 { color:#fff; font-size:2.35rem; margin:0; letter-spacing:-.5px; }
.hero p { color:#c7d8f7; margin:9px 0 0 0; }
div[data-testid="stMetric"] {
    background:#121c2e; border:1px solid #293956; padding:16px; border-radius:14px;
}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#e5eaf5; }
div[data-testid="stAlert"] { background:#102c26; border:1px solid #16734e; color:#b9f6d5; }
.stTextArea textarea {
    background:#111a2a !important; color:#edf3ff !important;
    border:1px solid #3156a5 !important; border-radius:12px !important;
}
.stTextArea textarea::placeholder { color:#8fa3c5 !important; }
.stButton > button {
    background:linear-gradient(90deg,#1769f5,#2454db); color:white;
    border:0; border-radius:11px; min-height:48px; font-weight:700;
}
.stButton > button:hover { background:linear-gradient(90deg,#3280ff,#416cf0); color:white; border:0; }
button[data-baseweb="tab"] { color:#aebed8; }
button[data-baseweb="tab"][aria-selected="true"] { color:#55a5ff; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background:#111a2a; border:1px solid #293956; border-radius:12px;
}
[data-testid="stExpander"] { background:#111a2a; border:1px solid #293956; border-radius:12px; }
.result-spam {
    background:#351923; border:1px solid #a83a55; color:#ff9cac;
    padding:18px; border-radius:14px; font-size:1.15rem; font-weight:700;
}
.result-ham {
    background:#102d27; border:1px solid #23845e; color:#8df0bd;
    padding:18px; border-radius:14px; font-size:1.15rem; font-weight:700;
}
hr { border-color:#293956; }
small, .stCaption { color:#91a4c4 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛡️ SpamGuard AI</h1>
  <p>SMS spam detection powered by Machine Learning and K-Nearest Neighbors (KNN)</p>
</div>
""", unsafe_allow_html=True)

DATA_URLS = [
    # Public mirror of the Kaggle SMS Spam Collection CSV.
    "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv",
    "https://raw.githubusercontent.com/kr0710/Data607/main/spamham.csv",
]

@st.cache_data(show_spinner=False)
def load_dataset():
    local_path = Path(__file__).parent / "data" / "spam.csv"
    if local_path.exists():
        return pd.read_csv(local_path, encoding="latin-1"), "Local project dataset"

    errors = []
    for url in DATA_URLS:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=20) as response:
                raw = response.read()
            frame = pd.read_csv(BytesIO(raw), encoding="latin-1")
            return frame, "Public copy of the Kaggle SMS Spam Collection dataset"
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("Could not automatically download the dataset. " + " | ".join(errors))

@st.cache_resource(show_spinner="Preparing text data and training the KNN model...")
def build_model(data_bytes):
    df = pd.read_csv(BytesIO(data_bytes), encoding="latin-1")
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    elif {"label", "message"}.issubset(df.columns):
        df = df[["label", "message"]]
    elif {"classification", "text_messages"}.issubset(df.columns):
        df = df[["classification", "text_messages"]].rename(
            columns={"classification": "label", "text_messages": "message"})
    else:
        raise ValueError("CSV columns not recognised. Expected Kaggle columns v1 and v2.")

    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["spam", "ham"])]
    df = df[df["message"].ne("")].drop_duplicates(subset=["message"])

    if len(df) < 20 or df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both spam and ham messages.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.20,
        random_state=42, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("knn", KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    report = classification_report(
        y_test, y_pred, labels=["ham", "spam"],
        target_names=["Not spam (ham)", "Spam"],
        output_dict=True, zero_division=0
    )
    return model, df, accuracy, matrix, report, len(X_train), len(X_test)

with st.sidebar:
    st.header("About this project")
    st.write("**Algorithm:** K-Nearest Neighbors")
    st.write("**Text features:** TF-IDF")
    st.write("**K value:** 5")
    st.write("**Split:** 80% training / 20% testing")
    st.divider()
    st.caption("Dataset source: Kaggle SMS Spam Collection Dataset.")
    st.caption("The app attempts to fetch the dataset automatically. Internet is required on first launch.")

try:
    raw_df, source_name = load_dataset()
    # Convert the loaded frame to bytes so the training function is cached by its data.
    data_bytes = raw_df.to_csv(index=False).encode("utf-8")
    model, df, accuracy, matrix, report, train_count, test_count = build_model(data_bytes)
except Exception as error:
    st.error("The app could not load/train on the dataset.")
    st.write(str(error))
    st.info("Check your internet connection and try again. You can also place the Kaggle file named spam.csv inside the project's data folder.")
    st.stop()

st.success(f"Dataset ready: {len(df):,} labelled messages loaded. Source: {source_name}")

predict_tab, performance_tab, dataset_tab = st.tabs(
    ["🔎 Try the detector", "📊 Model performance", "📁 Explore dataset"]
)

with predict_tab:
    st.subheader("Check a message")
    st.write("Paste an SMS or short text message below. The trained KNN model will compare it with examples in the dataset.")
    message = st.text_area(
        "Message text",
        placeholder="Example: Congratulations! You have won a free prize. Click here to claim now!",
        height=150
    )
    if st.button("Detect spam", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            prediction = model.predict([message])[0]
            if prediction == "spam":
                st.markdown('<div class="result-spam">⚠️ SPAM DETECTED</div>', unsafe_allow_html=True)
                st.write("This message is most similar to spam examples in the training dataset.")
            else:
                st.markdown('<div class="result-ham">✅ NOT SPAM (HAM)</div>', unsafe_allow_html=True)
                st.write("This message is most similar to legitimate examples in the training dataset.")
            st.caption("Machine-learning predictions can be wrong. Avoid opening suspicious links.")

with performance_tab:
    st.subheader("How well did the model perform?")
    one, two, three = st.columns(3)
    one.metric("Test accuracy", f"{accuracy * 100:.2f}%")
    two.metric("Training messages", f"{train_count:,}")
    three.metric("Testing messages", f"{test_count:,}")
    st.markdown("#### Confusion matrix")
    st.caption("Rows = actual class; columns = predicted class.")
    st.dataframe(
        pd.DataFrame(matrix, index=["Actual: Not spam", "Actual: Spam"],
                     columns=["Predicted: Not spam", "Predicted: Spam"]),
        use_container_width=True
    )
    st.markdown("#### Classification report")
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

with dataset_tab:
    st.subheader("Dataset overview")
    counts = df["label"].value_counts().rename(index={"ham": "Not spam", "spam": "Spam"})
    a, b = st.columns(2)
    a.metric("Total messages", f"{len(df):,}")
    b.metric("Spam messages", f"{int((df['label'] == 'spam').sum()):,}")
    st.bar_chart(counts)
    st.markdown("#### Sample messages")
    st.dataframe(df.head(12).rename(columns={"label": "Category", "message": "Message"}),
                 use_container_width=True)

st.divider()
st.caption("Built with Python, Streamlit, Pandas and Scikit-learn | KNN with cosine distance and K = 5")
st.caption("Academic note: Kaggle's SMS Spam Collection contains SMS messages, not full email records. This project demonstrates text-message spam classification.")
