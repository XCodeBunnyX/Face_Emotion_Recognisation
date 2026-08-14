"""
Real-Time Face Emotion Recognition Application
==============================================
Uses DeepFace and OpenCV to analyze real-time video stream from a webcam
and detect facial emotions dynamically.
"""

import argparse
import sys
import cv2
from deepface import DeepFace


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Face Emotion Recognition via OpenCV & DeepFace"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="opencv",
        choices=["opencv", "retinaface", "mtcnn", "ssd", "dlib", "mediapipe", "yolov8"],
        help="Face detector backend (default: opencv)"
    )
    parser.add_argument(
        "--enforce-detection",
        action="store_true",
        help="Enforce face detection exception if no face is found (default: False)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("==========================================")
    print(" Real-Time Face Emotion Recognition")
    print(" Press 'q' in the display window to quit.")
    print("==========================================")

    # Open video capture stream
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Error: Could not access video device index {args.device}.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video frame.")
            break

        try:
            # Perform facial emotion analysis
            results = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=args.enforce_detection,
                detector_backend=args.detector,
                silent=True
            )

            # Standardize list vs dict response from DeepFace
            if not isinstance(results, list):
                results = [results]

            for res in results:
                emotion = res.get("dominant_emotion", "Unknown")
                confidence = res.get("emotion", {}).get(emotion, 0.0)

                # Region coordinates if available
                region = res.get("region", {})
                x = region.get("x", 20)
                y = region.get("y", 40)
                w = region.get("w", 0)
                h = region.get("h", 0)

                # Draw bounding box if face location exists
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label_y = max(y - 10, 25)
                else:
                    label_y = 40

                # Display emotion text label with confidence score
                label = f"{emotion.capitalize()} ({confidence:.1f}%)"
                
                # Background rectangle for text contrast
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                cv2.rectangle(
                    frame,
                    (x, label_y - text_h - 5),
                    (x + text_w + 5, label_y + baseline),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        except Exception:
            # Silent fallback if frame cannot be processed or face not found
            pass

        # Display output video feed
        cv2.imshow("Real-Time Face Emotion Recognition", frame)

        # Exit loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting application...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()