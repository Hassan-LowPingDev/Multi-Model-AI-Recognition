import os

import cv2
import mediapipe as mp
import pandas as pd


# Dataset configuration
DATASET_DIR = os.path.join("Dataset", "256")
OUTPUT_FILE = "pose_dataset.csv"

SPLITS = ["training", "validation"]

CLASSES = [
    "bending",
    "lying",
    "sitting",
    "standing",
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5,
)


# Store extracted features
data = []

total_images = 0
successful_images = 0
skipped_images = 0


try:
    # Process training and validation separately
    for split in SPLITS:

        split_dir = os.path.join(DATASET_DIR, split)

        if not os.path.isdir(split_dir):
            print(f"Folder not found: {split_dir}")
            continue

        print(f"\n{'=' * 50}")
        print(f"Processing split: {split}")
        print(f"{'=' * 50}")

        # Process each class folder
        for label, class_name in enumerate(CLASSES):

            class_dir = os.path.join(
                split_dir,
                class_name,
            )

            if not os.path.isdir(class_dir):
                print(f"Class folder not found: {class_dir}")
                continue

            print(f"\nProcessing class: {class_name}")

            for filename in os.listdir(class_dir):

                if not filename.lower().endswith(
                    IMAGE_EXTENSIONS
                ):
                    continue

                image_path = os.path.join(
                    class_dir,
                    filename,
                )

                image = cv2.imread(image_path)

                if image is None:
                    print(f"Could not read: {image_path}")
                    skipped_images += 1
                    continue

                total_images += 1

                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )

                # Detect body landmarks
                results = pose_detector.process(image_rgb)

                if not results.pose_landmarks:
                    print(f"No pose detected: {image_path}")
                    skipped_images += 1
                    continue

                features = []

                # Extract 33 landmarks
                for landmark in results.pose_landmarks.landmark:

                    features.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility,
                    ])

                # Create a row containing the features
                row = {
                    f"feature_{i}": value
                    for i, value in enumerate(features)
                }

                # Store label and metadata
                row["label"] = label
                row["class_name"] = class_name
                row["filename"] = filename
                row["image_path"] = image_path.replace("\\", "/")
                row["split"] = split

                data.append(row)

                successful_images += 1

                print(f"Processed: {image_path}")

finally:
    pose_detector.close()


# Save extracted features
if not data:
    raise ValueError(
        "No pose landmarks were extracted. "
        "Check DATASET_DIR and the image folders."
    )

df = pd.DataFrame(data)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# Display summary
print("\n" + "=" * 50)
print("DATA PREPARATION COMPLETE")
print("=" * 50)

print(f"Images examined: {total_images}")
print(f"Successfully processed: {successful_images}")
print(f"Images skipped: {skipped_images}")
print(f"CSV saved to: {OUTPUT_FILE}")

print("\nImages per split and class:")

print(
    df.groupby(["split", "class_name"]).size()
)

print("\nTotal images per class:")

print(
    df["class_name"].value_counts()
)