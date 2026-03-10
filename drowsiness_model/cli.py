"""
cli.py — Command Line Interface
================================
Run drowsiness detection without the GUI.
Good for batch processing or headless environments.

Usage:
    python cli.py --image photo.jpg
    python cli.py --video dashcam.mp4 --show
    python cli.py --webcam
    python cli.py --image photo.jpg --output annotated.jpg --csv log.csv
"""

import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from utils.detector import DrowsinessDetector
from utils.logger import DrowsinessLogger
from config import EAR_THRESHOLD


def main():
    parser = argparse.ArgumentParser(
        description="⚠ Drowsiness Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --image dashcam.jpg
  python cli.py --image dashcam.jpg --output result.jpg --ear 0.22
  python cli.py --video footage.mp4 --show
  python cli.py --webcam --show
        """
    )

    parser.add_argument("--image",  type=str, help="Input image path")
    parser.add_argument("--video",  type=str, help="Input video path")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--ear",    type=float, default=EAR_THRESHOLD,
                        help=f"EAR threshold (default: {EAR_THRESHOLD})")
    parser.add_argument("--output", type=str, help="Save annotated image here")
    parser.add_argument("--csv",    type=str, help="Export detection log to CSV")
    parser.add_argument("--show",   action="store_true", help="Display output window")
    parser.add_argument("--no-age", action="store_true", help="Disable age estimation")

    args = parser.parse_args()

    if not any([args.image, args.video, args.webcam]):
        parser.print_help()
        print("\n❌ Provide --image, --video, or --webcam")
        sys.exit(1)

    detector = DrowsinessDetector()
    logger   = DrowsinessLogger()

    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Not found: {args.image}")
            sys.exit(1)

        result = detector.process_image(
            args.image, ear_threshold=args.ear, show_age=not args.no_age)

        people = result["people"]
        sleeping = [p for p in people if p["state"] == "sleeping"]
        logger.log(people, source=args.image)

        print(f"\n{'='*45}")
        print(f"  Total people : {len(people)}")
        print(f"  Sleeping     : {len(sleeping)}")
        print(f"  Awake        : {len(people) - len(sleeping)}")

        if sleeping:
            print(f"\n  ⚠️  SLEEPING:")
            for p in sleeping:
                age_s = f"  age ~{p['age']}" if p.get("age") else ""
                print(f"    🔴 Person {p['id']+1}  EAR={p['ear']:.3f}{age_s}")
        else:
            print("  ✓  All awake")
        print(f"{'='*45}\n")

        if args.output:
            cv2.imwrite(args.output, result["annotated"])
            print(f"Saved: {args.output}")

        if args.show:
            cv2.imshow("Drowsiness Detection", result["annotated"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif args.video or args.webcam:
        source = 0 if args.webcam else args.video
        if isinstance(source, str) and not os.path.exists(source):
            print(f"❌ Not found: {source}")
            sys.exit(1)

        print(f"Processing {'webcam' if args.webcam else source}...")
        print("Press Q to quit.\n")

        frame_n = [0]

        def on_frame(frame, people):
            frame_n[0] += 1
            sleeping = [p for p in people if p["state"] == "sleeping"]
            if frame_n[0] % 60 == 0:
                print(f"  Frame {frame_n[0]}: {len(people)} people, {len(sleeping)} sleeping")
            if sleeping:
                logger.log(people, source=str(source))

            if args.show:
                cv2.imshow("Drowsiness Detection — Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt()

        try:
            detector.process_video(
                source, ear_threshold=args.ear,
                show_age=not args.no_age, frame_cb=on_frame)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            cv2.destroyAllWindows()

    if args.csv:
        logger.export_csv(args.csv)
        summary = logger.get_summary()
        print(f"CSV exported: {args.csv}")
        print(f"Total logged: {summary['total']} detections, {summary['sleeping']} sleeping")


if __name__ == "__main__":
    main()
