from io import BytesIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "Image_recognition"
SOUND_DIR = ROOT / "SoundRecognition"
POSE_DIR = ROOT / "PoseRecognition"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_CLASSES = ["airplane", "automobile", "bird", "cat", "dog"]
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
POSE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, 5),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class SoundCNN(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, class_count),
        )

    def forward(self, features):
        return self.classifier(self.features(features))


@st.cache_resource
def load_image_model():
    model = ImageCNN().to(DEVICE)
    weights = torch.load(IMAGE_DIR / "image_recognition_model.pth", map_location=DEVICE, weights_only=True)
    if "model_state_dict" in weights:
        weights = weights["model_state_dict"]
    model.load_state_dict(weights)
    model.eval()
    return model


@st.cache_resource
def load_sound_model():
    checkpoint = torch.load(SOUND_DIR / "sound_model.pth", map_location=DEVICE, weights_only=True)
    classes = checkpoint.get("classes", ["dog", "rain", "siren", "car_horn", "clock_alarm"])
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


@st.cache_resource
def load_pose_model():
    with (POSE_DIR / "classes.json").open(encoding="utf-8") as file:
        classes = json.load(file)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    weights = torch.load(POSE_DIR / "pose_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(weights)
    model = model.to(DEVICE)
    model.eval()
    return model, classes


def image_prediction(image, model, transform, classes):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        scores = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    index = int(np.argmax(scores))
    return classes[index], scores


@st.cache_data
def sound_features(audio_bytes, sample_rate, duration, n_mels, n_fft, hop_length):
    import librosa

    audio, _ = librosa.load(BytesIO(audio_bytes), sr=sample_rate, mono=True, duration=duration)
    expected_length = sample_rate * duration
    if len(audio) < expected_length:
        audio = np.pad(audio, (0, expected_length - len(audio)))
    else:
        audio = audio[:expected_length]
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max) / 80.0
    return mel_db.astype(np.float32)


def sound_prediction(audio_bytes, model, settings):
    mel = sound_features(
        audio_bytes, settings["sample_rate"], settings["duration"],
        settings["n_mels"], settings["n_fft"], settings["hop_length"],
    )
    tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        scores = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    index = int(np.argmax(scores))
    return settings["classes"][index], scores


def show_scores(classes, scores):
    st.markdown("#### Class confidence")
    for name, score in zip(classes, scores):
        st.progress(float(score), text=f"{name.replace('_', ' ').title()} · {float(score):.1%}")


st.set_page_config(page_title="MultiModel AI Studio", page_icon="✦", layout="wide")
st.markdown("""
<style>
    :root {color-scheme: dark;}
    .stApp, [data-testid="stAppViewContainer"] {background: radial-gradient(circle at 85% 5%, #1c3650 0, #101827 40%, #0a101b 100%); color: #f2f6fc;}
    [data-testid="stHeader"] {background: transparent;}
    [data-testid="stSidebar"] {background: #101827 !important; border-right: 1px solid #2c415a;}
    [data-testid="stSidebar"] * {color: #dce7f5;}
    .block-container {max-width: 1250px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {padding: 1.7rem 2rem; border: 1px solid #2f4560; border-radius: 22px;
        background: linear-gradient(115deg, rgba(26,45,67,.96), rgba(19,31,49,.96)); margin-bottom: 1.2rem;}
    .hero small {color: #62ddc1; letter-spacing: .16em; text-transform: uppercase; font-weight: 700;}
    .hero h1 {margin: .45rem 0; color: #f7f9ff; font-size: 2.4rem;}
    .hero p {margin: 0; color: #bdc9da; font-size: 1.05rem;}
    [data-testid="stMetric"] {background: #152338; border: 1px solid #2c415a; padding: .9rem 1rem; border-radius: 15px;}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {color: #b8c8dc !important; opacity: 1 !important;}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {color: #f5f8ff !important;}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color: #aebed2 !important;}
    label, label p, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {color: #e0e8f3;}
    [data-testid="stProgressBar"] {background: #26374d;}
    [data-testid="stProgressBar"] > div {background: #4fd1b5;}
    @keyframes reveal-result {
        from {opacity: 0; transform: translateY(9px);}
        to {opacity: 1; transform: translateY(0);}
    }
    [data-testid="stMetric"], [data-testid="stAlert"], .stProgress {
        animation: reveal-result .45s ease-out both;
    }
    @media (prefers-reduced-motion: reduce) {
        [data-testid="stMetric"], [data-testid="stAlert"], .stProgress {animation: none;}
    }
    div[data-testid="stTabs"] button {font-weight: 650;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <small>Applied AI · Interactive model portfolio</small>
  <h1>MultiModel AI Studio</h1>
  <p>Explore three trained models for image, sound, and human posture recognition.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Model lab")
    selected_model = st.radio(
        "Choose a model to test",
        ["Image Recognition", "Sound Recognition", "Human Pose Recognition"],
    )
    st.divider()
    st.caption(f"Inference device: **{DEVICE.type.upper()}**")
    st.caption("Models load only when selected and stay cached for fast repeat predictions.")


if selected_model == "Image Recognition":
    st.subheader("Image recognition")
    st.write("Classify airplanes, automobiles, birds, cats, and dogs.")
    model_path = IMAGE_DIR / "image_recognition_model.pth"
    if not model_path.exists():
        st.error("Image model not found. Add Image_recognition/image_recognition_model.pth.")
        st.stop()
    model = load_image_model()
    tabs = st.tabs(["Provided test images", "Upload an image"])
    with tabs[0]:
        image_files = sorted((IMAGE_DIR / "test_images").glob("*.png"))
        answer_path = IMAGE_DIR / "answer_key.csv"
        answer_key = {}
        if answer_path.exists():
            answer_key = pd.read_csv(answer_path).set_index("filename")["actual_class"].to_dict()
        if image_files:
            if answer_key:
                categories = sorted({answer_key[path.name] for path in image_files if path.name in answer_key})
                selected_category = st.selectbox(
                    "1 · Choose an image category",
                    categories,
                    format_func=lambda name: name.title(),
                    help="Choose a class first to narrow the image list.",
                )
                filtered_images = [path for path in image_files if answer_key.get(path.name) == selected_category]
                st.caption(f"Showing {len(filtered_images)} test images labeled {selected_category.title()}.")
            else:
                filtered_images = image_files
                st.caption("Choose a test image. Add answer_key.csv to filter these images by class.")
            chosen = st.selectbox(
                "2 · Choose a test image",
                filtered_images,
                format_func=lambda path: path.name,
                help="You can type a filename to jump directly to it.",
            )
            image = Image.open(chosen).convert("RGB")
            prediction, scores = image_prediction(image, model, IMAGE_TRANSFORM, IMAGE_CLASSES)
            left, right = st.columns([1, 1.1])
            with left:
                st.image(image, caption=chosen.name, use_container_width=True)
            with right:
                st.metric("Predicted class", prediction.title())
                st.metric("Confidence", f"{float(np.max(scores)):.1%}")
                actual = answer_key.get(chosen.name)
                if actual:
                    st.write(f"Dataset label: **{actual.title()}**")
                    st.success("Correct prediction" if actual == prediction else "Prediction differs from the label")
                show_scores(IMAGE_CLASSES, scores)
        else:
            st.info("No bundled test images found. Use the upload tab to try an image.")
    with tabs[1]:
        uploaded = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg", "webp"], key="image_upload")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            prediction, scores = image_prediction(image, model, IMAGE_TRANSFORM, IMAGE_CLASSES)
            left, right = st.columns([1, 1.1])
            left.image(image, caption=uploaded.name, use_container_width=True)
            right.metric("Predicted class", prediction.title())
            right.metric("Confidence", f"{float(np.max(scores)):.1%}")
            show_scores(IMAGE_CLASSES, scores)


elif selected_model == "Sound Recognition":
    st.subheader("Sound recognition")
    st.write("Listen to each clip while the model identifies dog, rain, siren, car horn, or clock alarm sounds.")
    model_path = SOUND_DIR / "sound_model.pth"
    audio_dir = SOUND_DIR / "ESC-50" / "audio"
    csv_path = SOUND_DIR / "ESC-50" / "meta" / "esc50.csv"
    if not model_path.exists():
        st.error("Sound model not found. Add SoundRecognition/sound_model.pth.")
        st.stop()
    model, settings = load_sound_model()
    tabs = st.tabs(["Held-out test sounds", "Upload a sound"])
    with tabs[0]:
        if not audio_dir.exists() or not csv_path.exists():
            st.info("ESC-50 audio or metadata is unavailable. Upload a sound in the next tab.")
        else:
            metadata = pd.read_csv(csv_path)
            metadata = metadata[(metadata["category"].isin(settings["classes"])) & (metadata["fold"] == 5)]
            if metadata.empty:
                st.info("No held-out test sounds were found for these classes.")
            else:
                class_filter = st.selectbox(
                    "1 · Choose a sound category",
                    settings["classes"],
                    format_func=lambda name: name.replace("_", " ").title(),
                    help="Select a sound class to see only its held-out test clips.",
                )
                visible = metadata[metadata["category"] == class_filter]
                st.caption(f"Showing {len(visible)} test clips for {class_filter.replace('_', ' ').title()}.")
                selected = st.selectbox(
                    "2 · Choose a sound clip",
                    visible["filename"].tolist(),
                    help="Select a clip to listen to it and view the model prediction.",
                )
                row = metadata.loc[metadata["filename"] == selected].iloc[0]
                audio_bytes = (audio_dir / selected).read_bytes()
                run_sound_evaluation = False
                left, right = st.columns([1, 1.1])
                with left:
                    st.markdown("#### Listen to the clip")
                    st.audio(audio_bytes, format="audio/wav")
                    st.caption(f"{selected} · ESC-50 held-out fold {int(row['fold'])}")
                    with st.expander("Evaluate all held-out test sounds"):
                        st.caption("Uses fold 5, reserved as the test split by train.py.")
                        run_sound_evaluation = st.button("Run sound test evaluation", type="primary")
                with right:
                    prediction, scores = sound_prediction(audio_bytes, model, settings)
                    st.metric("Predicted sound", prediction.replace("_", " ").title())
                    st.metric("Confidence", f"{float(np.max(scores)):.1%}")
                    actual = row["category"]
                    st.write(f"Dataset label: **{actual.replace('_', ' ').title()}**")
                    st.success("Correct prediction" if prediction == actual else "Prediction differs from the label")
                    show_scores(settings["classes"], scores)

                if run_sound_evaluation:
                    results = []
                    progress = st.progress(0, text="Analyzing held-out clips…")
                    for index, (_, test_row) in enumerate(metadata.iterrows(), start=1):
                        test_file = audio_dir / test_row["filename"]
                        test_bytes = test_file.read_bytes()
                        result, probabilities = sound_prediction(test_bytes, model, settings)
                        results.append({
                            "Recording": test_row["filename"],
                            "Actual": test_row["category"].replace("_", " ").title(),
                            "Predicted": result.replace("_", " ").title(),
                            "Confidence": f"{float(np.max(probabilities)):.1%}",
                            "Correct": result == test_row["category"],
                        })
                        progress.progress(index / len(metadata), text=f"Analyzed {index} of {len(metadata)} clips")
                    accuracy = sum(result["Correct"] for result in results) / len(results)
                    st.metric("Held-out accuracy", f"{accuracy:.1%}",
                              help=f"{sum(result['Correct'] for result in results)} of {len(results)} correct")
                    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
    with tabs[1]:
        uploaded = st.file_uploader("Upload an audio clip", type=["wav", "mp3", "ogg", "flac", "m4a"], key="sound_upload")
        if uploaded:
            audio_bytes = uploaded.getvalue()
            st.audio(audio_bytes)
            try:
                prediction, scores = sound_prediction(audio_bytes, model, settings)
                left, right = st.columns([1, 1.2])
                left.metric("Recognized sound", prediction.replace("_", " ").title())
                right.metric("Confidence", f"{float(np.max(scores)):.1%}")
                show_scores(settings["classes"], scores)
            except Exception as error:
                st.error(f"Could not read this audio. Try a WAV, MP3, OGG, FLAC, or M4A file. Details: {error}")


else:
    st.subheader("Human pose recognition")
    st.write("Classify a person’s posture as bending, lying, sitting, or standing.")
    model_path = POSE_DIR / "pose_model.pth"
    classes_path = POSE_DIR / "classes.json"
    validation_dir = POSE_DIR / "dataSetFolder" / "256" / "validation"
    if not model_path.exists() or not classes_path.exists():
        st.error("Pose model or classes.json is missing from PoseRecognition.")
        st.stop()
    model, classes = load_pose_model()
    tabs = st.tabs(["Validation examples", "Upload a photo"])
    with tabs[0]:
        samples = []
        for class_name in classes:
            class_dir = validation_dir / class_name
            if class_dir.exists():
                samples.extend((path, class_name) for path in sorted(class_dir.iterdir())
                               if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if samples:
            selected_class = st.selectbox(
                "1 · Choose a posture category",
                classes,
                format_func=lambda name: name.title(),
                help="Choose a posture first to narrow the validation image list.",
            )
            class_samples = [item for item in samples if item[1] == selected_class]
            st.caption(f"Showing {len(class_samples)} validation images labeled {selected_class.title()}.")
            selected = st.selectbox(
                "2 · Choose a validation image",
                class_samples,
                format_func=lambda item: item[0].name,
                help="You can type a filename to find a particular example.",
            )
            image = Image.open(selected[0]).convert("RGB")
            prediction, scores = image_prediction(image, model, POSE_TRANSFORM, classes)
            left, right = st.columns([1, 1.1])
            with left:
                st.image(image, caption=f"Validation label: {selected[1].title()}", use_container_width=True)
            with right:
                st.metric("Predicted posture", prediction.title())
                st.metric("Confidence", f"{float(np.max(scores)):.1%}")
                st.write(f"Dataset label: **{selected[1].title()}**")
                st.success("Correct prediction" if prediction == selected[1] else "Prediction differs from the label")
                show_scores(classes, scores)
        else:
            st.info("Validation images are not available locally. Upload a photo in the next tab.")
    with tabs[1]:
        uploaded = st.file_uploader("Upload a full-body posture photo", type=["png", "jpg", "jpeg", "webp"], key="pose_upload")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            prediction, scores = image_prediction(image, model, POSE_TRANSFORM, classes)
            left, right = st.columns([1, 1.1])
            left.image(image, caption=uploaded.name, use_container_width=True)
            right.metric("Predicted posture", prediction.title())
            right.metric("Confidence", f"{float(np.max(scores)):.1%}")
            show_scores(classes, scores)
