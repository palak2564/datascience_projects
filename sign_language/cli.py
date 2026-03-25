"""
cli.py — Sign Language Detection CLI
======================================
Run sign detection from the command line (Colab-friendly).

Usage:
    python cli.py --image hand.jpg
    python cli.py --image hand.jpg --demo   # skip time window
    python cli.py --video hand_video.mp4
    python cli.py --webcam
"""

import argparse
import os
import sys
import cv2
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.detector import SignLanguageDetector
from utils.logger import PredictionLogger
from config import ACTIVE_START, ACTIVE_END


def is_active():
    from datetime import datetime
    now = datetime.now().time()
    return ACTIVE_START <= now <= ACTIVE_END


def main():
    parser = argparse.ArgumentParser(description="◈ Sign Language Detection CLI")
    parser.add_argument("--image",  type=str, help="Input image path")
    parser.add_argument("--video",  type=str, help="Input video path")
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--conf",   type=float, default=0.60)
    parser.add_argument("--output", type=str, help="Save annotated image")
    parser.add_argument("--csv",    type=str, help="Export CSV log")
    parser.add_argument("--demo",   action="store_true",
                        help="Skip time window check")
    args = parser.parse_args()

    if not args.demo and not is_active():
        from datetime import datetime
        print(f"\n⏰  System is only active {ACTIVE_START.strftime('%I:%M %p')} – "
              f"{ACTIVE_END.strftime('%I:%M %p')}")
        print(f"   Current time: {datetime.now().strftime('%I:%M %p')}")
        print("   Use --demo to bypass for testing.\n")
        sys.exit(0)

    if not any([args.image, args.video, args.webcam]):
        parser.print_help()
        sys.exit(1)

    detector = SignLanguageDetector()
    logger   = PredictionLogger()
    os.makedirs("outputs", exist_ok=True)

    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Not found: {args.image}")
            sys.exit(1)

        result = detector.predict_image(args.image, min_confidence=args.conf)
        pred = result.get("prediction", "—")
        conf = result.get("confidence", 0.0)
        hands = result.get("hands_detected", 0)

        print(f"\n{'═'*40}")
        print(f"  Image    : {os.path.basename(args.image)}")
        print(f"  Sign     : {pred}")
        print(f"  Conf     : {conf:.1%}")
        print(f"  Hands    : {hands}")
        print(f"{'═'*40}\n")

        logger.log(pred, conf, source=args.image, mode="image")

        if args.output and result.get("annotated") is not None:
            cv2.imwrite(args.output, result["annotated"])
            print(f"✓ Saved: {args.output}")

    elif args.video or args.webcam:
        source = 0 if args.webcam else args.video
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Cannot open: {source}")
            sys.exit(1)

        print(f"Processing {'webcam' if args.webcam else source}...")
        print("Press Q to quit.\n")
        frame_n = 0
        last_pred_t = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1

            now = time.time()
            if now - last_pred_t >= 0.5:
                result = detector.predict_frame(frame, min_confidence=args.conf)
                pred = result.get("prediction")
                conf = result.get("confidence", 0.0)
                if pred:
                    print(f"  [{frame_n:5d}]  {pred:16s}  {conf:.1%}")
                    logger.log(pred, conf, source=str(source), mode="webcam")
                last_pred_t = now

            annotated = detector.draw_landmarks(frame.copy())
            cv2.imshow("Sign Language — Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    if args.csv:
        logger.export_csv(args.csv)
        summary = logger.get_summary()
        print(f"\n✓ CSV: {args.csv}")
        print(f"  Total predictions: {summary['total']}")
        print(f"  Top signs: {summary['top_signs']}")


if __name__ == "__main__":
    main()
