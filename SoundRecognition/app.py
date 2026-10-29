from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "ESC-50" / "audio"
CSV_PATH = BASE_DIR / "ESC-50" / "meta" / "esc50.csv"
MODEL_PATH = BASE_DIR / "sound_model.pth"
DEFAULT_CLASSES = ["dog", "rain", "siren", "car_horn", "clock_alarm"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoundCNN(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, class_count),
        )

    def forward(self, audio_features):
        return self.classifier(self.features(audio_features))


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    classes = checkpoint.get("classes", DEFAULT_CLASSES)
    model = SoundCNN(len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    settings = {
        "classes": classes,
        "sample_rate": checkpoint.get("sample_rate", 22050),
        "n_mels": checkpoint.get("n_mels", 64),
        "n_fft": checkpoint.get("n_fft", 1024),
        "hop_length": checkpoint.get("hop_length", 512),
        "duration": checkpoint.get("duration", 5),
    }
    return model, settings


@st.cache_data
def load_test_metadata(classes):
    metadata = pd.read_csv(CSV_PATH)
    return metadata[(metadata["category"].isin(classes)) & (metadata["fold"] == 5)].copy()


def extract_features(audio_source, settings):
    audio, _ = librosa.load(
        audio_source,
        sr=settings["sample_rate"],
        mono=True,
        duration=settings["duration"],
    )
    expected_length = settings["sample_rate"] * settings["duration"]
    if len(audio) < expected_length:
        audio = np.pad(audio, (0, expected_length - len(audio)))
    else:
        audio = audio[:expected_length]
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=settings["sample_rate"],
        n_fft=settings["n_fft"],
        hop_length=settings["hop_length"],
        n_mels=settings["n_mels"],
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max) / 80.0
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def predict(audio_source, model, settings):
    features = extract_features(audio_source, settings).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(features), dim=1)[0].cpu().numpy()
    best_index = int(np.argmax(probabilities))
    return settings["classes"][best_index], probabilities


st.set_page_config(page_title="Sound Recognition Studio", page_icon="🎙️", layout="wide")
st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #0b1020 0%, #101a2f 55%, #17243a 100%); color: #f4f7fb;}
    [data-testid="stHeader"] {background: transparent;}
    .hero {padding: 1.6rem 1.8rem; border: 1px solid #33435e; border-radius: 20px;
        background: linear-gradient(110deg, #17243a, #17233a 55%, #263b56); margin-bottom: 1rem;}
    .hero small {color: #8ee3d1; letter-spacing: .14em; text-transform: uppercase; font-weight: 700;}
    .hero h1 {margin: .35rem 0; color: #f5f8ff; font-size: 2.25rem;}
    .hero p {margin: 0; color: #bfccdf;}
    [data-testid="stMetric"] {background: #18253a; border: 1px solid #30415b; padding: 1rem; border-radius: 14px;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <small>Audio intelligence · ESC-50</small>
  <h1>Sound Recognition Studio</h1>
  <p>Listen to a clip, inspect the model’s confidence, and explore its held-out test results.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()
if not CSV_PATH.exists() or not AUDIO_DIR.exists():
    st.error("The ESC-50 metadata or audio folder could not be found.")
    st.stop()

try:
    model, settings = load_model()
    metadata = load_test_metadata(tuple(settings["classes"]))
except Exception as error:
    st.error(f"Could not prepare the model and test set: {error}")
    st.stop()

st.caption(f"Model ready · {len(settings['classes'])} sound classes · {DEVICE.type.upper()} inference")
tab_test, tab_upload = st.tabs(["Held-out test set", "Try your own sound"])

with tab_test:
    if metadata.empty:
        st.warning("No fold 5 test recordings were found for the model’s classes.")
    else:
        class_filter = st.selectbox("Filter by sound class", ["All classes"] + settings["classes"])
        visible_metadata = metadata if class_filter == "All classes" else metadata[metadata["category"] == class_filter]
        chosen = st.selectbox(
            "Choose a test recording",
            visible_metadata["filename"].tolist(),
            format_func=lambda filename: f"{filename} · {visible_metadata.loc[visible_metadata['filename'] == filename, 'category'].iloc[0].replace('_', ' ').title()}",
        )
        row = metadata.loc[metadata["filename"] == chosen].iloc[0]
        audio_path = AUDIO_DIR / chosen
        audio_bytes = audio_path.read_bytes()
        actual_class = row["category"]

        left, right = st.columns([1.05, 1])
        with left:
            st.subheader("Listen to the recording")
            st.audio(audio_bytes, format="audio/wav")
            st.caption(f"{chosen} · 5-second ESC-50 clip · Test fold {int(row['fold'])}")
        with right:
            st.subheader("Model prediction")
            with st.spinner("Analyzing the sound…"):
                predicted_class, probabilities = predict(str(audio_path), model, settings)
            confidence = float(probabilities[settings["classes"].index(predicted_class)])
            metric_a, metric_b = st.columns(2)
            metric_a.metric("Predicted sound", predicted_class.replace("_", " ").title())
            metric_b.metric("Confidence", f"{confidence:.1%}")
            st.write(f"**Dataset label:** {actual_class.replace('_', ' ').title()}")
            st.success("Prediction matches the dataset label" if predicted_class == actual_class else "Prediction differs from the dataset label")
            chart_data = pd.DataFrame({"Class": [name.replace("_", " ").title() for name in settings["classes"]],
                                       "Confidence": probabilities})
            st.bar_chart(chart_data, x="Class", y="Confidence", color="#58d6be")

        with st.expander("Evaluate the complete held-out test fold"):
            st.write("This evaluation uses ESC-50 fold 5, which `train.py` reserves for testing.")
            if st.button("Run test fold evaluation", type="primary"):
                rows = []
                progress = st.progress(0, text="Analyzing test recordings…")
                for index, (_, test_row) in enumerate(metadata.iterrows(), start=1):
                    file_path = AUDIO_DIR / test_row["filename"]
                    prediction, scores = predict(str(file_path), model, settings)
                    rows.append({"Recording": test_row["filename"],
                                 "Actual class": test_row["category"].replace("_", " ").title(),
                                 "Predicted class": prediction.replace("_", " ").title(),
                                 "Confidence": f"{float(np.max(scores)):.1%}",
                                 "Correct": prediction == test_row["category"]})
                    progress.progress(index / len(metadata), text=f"Analyzed {index} of {len(metadata)} recordings")
                correct_count = sum(item["Correct"] for item in rows)
                st.metric("Held-out test accuracy", f"{correct_count / len(rows):.1%}",
                          help=f"{correct_count} correct out of {len(rows)} recordings")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_upload:
    uploaded_audio = st.file_uploader("Upload a sound clip", type=["wav", "mp3", "ogg", "flac", "m4a"])
    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.getvalue()
        st.audio(audio_bytes)
        try:
            predicted_class, probabilities = predict(BytesIO(audio_bytes), model, settings)
            confidence = float(probabilities[settings["classes"].index(predicted_class)])
            left, right = st.columns([1, 1.2])
            left.metric("Recognized sound", predicted_class.replace("_", " ").title())
            right.metric("Confidence", f"{confidence:.1%}")
            chart_data = pd.DataFrame({"Class": [name.replace("_", " ").title() for name in settings["classes"]],
                                       "Confidence": probabilities})
            st.bar_chart(chart_data, x="Class", y="Confidence", color="#58d6be")
        except Exception as error:
            st.error(f"Could not read this audio file. Try a WAV, MP3, OGG, FLAC, or M4A recording. Details: {error}")
