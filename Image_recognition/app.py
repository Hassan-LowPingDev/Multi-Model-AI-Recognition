from pathlib import Path
import csv

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "test_images"
MODEL_PATH = BASE_DIR / "image_recognition_model.pth"
CLASSES = ["airplane", "automobile", "bird", "cat", "dog"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 5),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


@st.cache_resource
def load_model():
    model = ImageCNN().to(DEVICE)
    weights = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in weights:
        weights = weights["model_state_dict"]
    model.load_state_dict(weights)
    model.eval()
    return model


@st.cache_data
def load_answer_key():
    answer_key = {}
    answer_path = BASE_DIR / "answer_key.csv"
    if answer_path.exists():
        with answer_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                answer_key[row["filename"]] = row["actual_class"]
    return answer_key


TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def predict(image, model):
    image_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    confidence, class_index = probabilities.max(dim=0)
    return CLASSES[class_index.item()], confidence.item(), probabilities.cpu().tolist()


st.set_page_config(page_title="Image Recognition Tester", page_icon="🖼️", layout="wide")
st.title("Image Recognition Model Tester")
st.write("Test the trained five class model with the provided images or upload your own.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

st.caption(f"Running on {DEVICE.type.upper()} · Classes: {', '.join(CLASSES)}")
tab_test, tab_upload = st.tabs(["Provided test images", "Upload an image"])

with tab_test:
    image_files = sorted(IMAGE_DIR.glob("*")) if IMAGE_DIR.exists() else []
    image_files = [path for path in image_files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if not image_files:
        st.info("No test images found in the test_images folder.")
    else:
        answer_key = load_answer_key()
        selected_name = st.selectbox("Choose a test image", [path.name for path in image_files])
        selected_path = IMAGE_DIR / selected_name
        image = Image.open(selected_path).convert("RGB")
        prediction, confidence, probabilities = predict(image, model)
        actual_class = answer_key.get(selected_name)

        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption=selected_name, use_container_width=True)
        with right:
            st.subheader("Prediction")
            st.metric("Predicted class", prediction)
            st.metric("Confidence", f"{confidence:.1%}")
            if actual_class:
                st.write(f"**Actual class:** {actual_class}")
                st.success("Correct prediction" if prediction == actual_class else "Incorrect prediction")
            st.write("Class probabilities")
            for label, probability in zip(CLASSES, probabilities):
                st.progress(probability, text=f"{label}: {probability:.1%}")

        with st.expander("Evaluate all provided images"):
            if answer_key:
                results = []
                for path in image_files:
                    image = Image.open(path).convert("RGB")
                    predicted, score, _ = predict(image, model)
                    actual = answer_key.get(path.name, "")
                    results.append({"Image": path.name, "Actual": actual, "Predicted": predicted,
                                    "Confidence": f"{score:.1%}", "Correct": predicted == actual if actual else ""})
                labeled_results = [row for row in results if row["Actual"]]
                if labeled_results:
                    accuracy = sum(row["Correct"] for row in labeled_results) / len(labeled_results)
                    st.metric("Accuracy on labeled test images", f"{accuracy:.1%}",
                              help=f"{sum(row['Correct'] for row in labeled_results)} of {len(labeled_results)} correct")
                st.dataframe(results, use_container_width=True, hide_index=True)
            else:
                st.info("answer_key.csv is missing or empty, so accuracy cannot be calculated.")

with tab_upload:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        prediction, confidence, probabilities = predict(image, model)
        left, right = st.columns(2)
        with left:
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        with right:
            st.subheader(f"Prediction: {prediction}")
            st.metric("Confidence", f"{confidence:.1%}")
            for label, probability in zip(CLASSES, probabilities):
                st.progress(probability, text=f"{label}: {probability:.1%}")
