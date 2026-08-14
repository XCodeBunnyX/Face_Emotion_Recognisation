"""
Static Image Face Emotion Recognition Script
============================================
Analyzes images to detect faces and classify dominant emotions.
Saves and/or displays the annotated output image.
"""

import argparse
import sys
import os
import cv2
from deepface import DeepFace


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze static images for facial emotion recognition."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="samples/happy_boy.jpg",
        help="Path to input image file (default: samples/happy_boy.jpg)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output image (optional)"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="opencv",
        choices=["opencv", "retinaface", "mtcnn", "ssd", "dlib", "mediapipe", "yolov8"],
        help="Face detector backend (default: opencv)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable interactive OpenCV display window"
    )
    return parser.parse_args()


def analyze_image(image_path, output_path=None, detector="opencv", display=True):
    if not os.path.exists(image_path):
        print(f"Error: File not found at path '{image_path}'")
        return False

    print(f"Analyzing image: {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image file '{image_path}'")
        return False

    try:
        results = DeepFace.analyze(
            img_path=img,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=detector,
            silent=True
        )

        if not isinstance(results, list):
            results = [results]

        print(f"\nDetected {len(results)} face(s):")

        for idx, res in enumerate(results, 1):
            dominant_emotion = res.get("dominant_emotion", "Unknown")
            emotions = res.get("emotion", {})
            confidence = emotions.get(dominant_emotion, 0.0)

            print(f" Face #{idx}: {dominant_emotion.capitalize()} ({confidence:.2f}%)")
            print("  Full Emotion Scores:")
            for emo, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {emo.capitalize():<10}: {score:.2f}%")

            # Draw bounding box and text annotation
            region = res.get("region", {})
            x, y, w, h = region.get("x", 20), region.get("y", 40), region.get("w", 0), region.get("h", 0)

            if w > 0 and h > 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label_y = max(y - 10, 25)
            else:
                label_y = 40

            label = f"{dominant_emotion.capitalize()} ({confidence:.1f}%)"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                img,
                (x, label_y - text_h - 5),
                (x + text_w + 5, label_y + baseline),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                img,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        if output_path:
            cv2.imwrite(output_path, img)
            print(f"\nSaved annotated result to: {output_path}")

        if display:
            cv2.imshow("Emotion Recognition Output", img)
            print("\nPress any key in the image window to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return True

    except Exception as e:
        print(f"Error during emotion analysis: {e}")
        return False


def main():
    args = parse_args()
    analyze_image(
        image_path=args.image,
        output_path=args.output,
        detector=args.detector,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
