"""
cli.py — Command Line Interface
================================
Run the animal detector from the command line without opening the GUI.
Useful for batch processing or testing on a headless server.

Usage:
    python cli.py --image path/to/image.jpg
    python cli.py --video path/to/video.mp4
    python cli.py --image animals.jpg --conf 0.5 --model yolov8s
    python cli.py --image animals.jpg --output result.jpg

Why have a CLI? Because:
1. Faster to test without GUI overhead
2. Easier to batch-process folders of images
3. Good for running on servers without a display
"""

import argparse
import os
import sys
import cv2

# Make sure we can import from this project
sys.path.insert(0, os.path.dirname(__file__))

from utils.detector import AnimalDetector
from utils.logger import DetectionLogger


def main():
    parser = argparse.ArgumentParser(
        description="🐾 Animal Detection CLI — detect and classify animals in images/videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --image zoo.jpg
  python cli.py --image zoo.jpg --conf 0.6 --output annotated.jpg
  python cli.py --video wildlife.mp4 --model yolov8s
  python cli.py --image zoo.jpg --no-labels
        """
    )

    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam (device 0)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--model", type=str, default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
                        help="Model size (default: yolov8n)")
    parser.add_argument("--output", type=str, help="Save annotated image to this path")
    parser.add_argument("--no-labels", action="store_true", help="Hide text labels on boxes")
    parser.add_argument("--export-csv", type=str, help="Export detection log to CSV file")
    parser.add_argument("--show", action="store_true", help="Display result in a window (requires display)")

    args = parser.parse_args()

    if not any([args.image, args.video, args.webcam]):
        parser.print_help()
        print("\n❌ Please provide --image, --video, or --webcam")
        sys.exit(1)

    # Init
    print(f"[cli] Loading model: {args.model}")
    detector = AnimalDetector(model_size=args.model)
    logger = DetectionLogger()

    # ── Image mode ──────────────────────────────────────────────────────────

    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)

        print(f"[cli] Processing image: {args.image}")
        result = detector.detect_image(
            args.image,
            confidence=args.conf,
            show_labels=not args.no_labels,
        )

        detections = result["detections"]
        logger.log(detections, source=args.image)

        # Print summary
        print(f"\n{'='*40}")
        print(f"  Animals detected: {len(detections)}")
        carnivores = [d for d in detections if d["is_carnivore"]]
        if carnivores:
            print(f"  ⚠️  CARNIVORES ({len(carnivores)}):")
            for c in carnivores:
                print(f"      🔴 {c['label'].title()} — {c['confidence']:.0%}")
        else:
            print("  ✓ No carnivores detected")

        print("\n  Full list:")
        for d in detections:
            icon = "🔴" if d["is_carnivore"] else "🟢"
            print(f"    {icon} {d['label'].title():15s} {d['confidence']:.0%}  [{d['diet_tag']}]")
        print(f"{'='*40}\n")

        # Save / show
        if args.output:
            cv2.imwrite(args.output, result["annotated_frame"])
            print(f"[cli] Saved annotated image: {args.output}")

        if args.show:
            cv2.imshow("Animal Detection", result["annotated_frame"])
            print("[cli] Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # ── Video / webcam mode ─────────────────────────────────────────────────

    elif args.video or args.webcam:
        source = 0 if args.webcam else args.video

        if args.video and not os.path.exists(args.video):
            print(f"❌ Video not found: {args.video}")
            sys.exit(1)

        print(f"[cli] Processing {'webcam' if args.webcam else args.video}")
        print("[cli] Press Q to quit")

        frame_count = 0

        def on_frame(frame, detections):
            nonlocal frame_count
            frame_count += 1

            if args.show:
                cv2.imshow("Animal Detection — press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt()

            if frame_count % 30 == 0:
                carnivores = [d for d in detections if d["is_carnivore"]]
                print(f"  Frame {frame_count}: {len(detections)} animals, {len(carnivores)} carnivores")

            logger.log(detections, source=str(source))

        try:
            detector.detect_video(source, confidence=args.conf, frame_callback=on_frame)
        except KeyboardInterrupt:
            print("\n[cli] Stopped by user")
        finally:
            cv2.destroyAllWindows()

        print(f"\n[cli] Total frames processed: {frame_count}")
        summary = logger.get_summary()
        print(f"[cli] Total detections logged: {summary['total']}")

    # ── Export ──────────────────────────────────────────────────────────────

    if args.export_csv:
        logger.export_csv(args.export_csv)
        print(f"[cli] Detection log exported: {args.export_csv}")


if __name__ == "__main__":
    main()
