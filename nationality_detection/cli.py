"""
cli.py — Command Line Interface
================================
Run nationality detection without the GUI.
Perfect for Colab, headless servers, or batch processing.

Usage:
    python cli.py --image photo.jpg
    python cli.py --image photo.jpg --output result.jpg --csv log.csv
    python cli.py --batch samples/
"""

import argparse
import os
import sys
import cv2
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.predictor import NationalityPredictor
from utils.logger import PredictionLogger


def print_result(result: dict):
    """Pretty-print a prediction result to console."""
    print(f"\n{'═'*48}")
    print(f"  IMAGE: {os.path.basename(result.get('image_path', '?'))}")
    print(f"{'─'*48}")

    grp = result.get("task_group", "other")
    nat = result.get("nationality", "Unknown")
    conf = result.get("confidence", 0)
    print(f"  Nationality  : {nat}  ({conf:.1f}% confidence)")
    print(f"  Task Group   : {grp.upper()}")
    print(f"  Face Found   : {'Yes ✓' if result.get('face_found') else 'No ✗'}")
    print(f"{'─'*48}")

    preds = result.get("predictions_made", [])
    if "emotion" in preds:
        print(f"  Emotion      : {result.get('emotion', 'N/A')}")
    if "age" in preds and result.get("age"):
        print(f"  Age          : ~{result['age']} years")
    if "age_bracket" in preds and result.get("age_bracket") and not result.get("age"):
        print(f"  Age Range    : {result['age_bracket']}")
    if "dress_colour" in preds:
        print(f"  Dress Colour : {result.get('dress_colour', 'N/A')}")

    print(f"{'═'*48}\n")


def main():
    parser = argparse.ArgumentParser(
        description="🌐 Nationality Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --image face.jpg
  python cli.py --image face.jpg --output annotated.jpg
  python cli.py --image face.jpg --json
  python cli.py --batch samples/
  python cli.py --image face.jpg --csv results.csv
        """
    )
    parser.add_argument("--image",  type=str, help="Input image path")
    parser.add_argument("--batch",  type=str, help="Folder of images to process")
    parser.add_argument("--output", type=str, help="Save annotated image here")
    parser.add_argument("--csv",    type=str, help="Export log to CSV")
    parser.add_argument("--json",   action="store_true", help="Print result as JSON")

    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.print_help()
        sys.exit(1)

    predictor = NationalityPredictor()
    logger    = PredictionLogger()

    os.makedirs("outputs", exist_ok=True)

    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ File not found: {args.image}")
            sys.exit(1)

        print(f"\nAnalysing: {args.image}")
        result = predictor.analyse(args.image)
        logger.log(result, source=args.image)

        if args.json:
            # Exclude non-serialisable items
            safe = {k: v for k, v in result.items()
                    if k != "annotated_image" and v is not None}
            print(json.dumps(safe, indent=2))
        else:
            print_result(result)

        if args.output and result.get("annotated_image") is not None:
            cv2.imwrite(args.output, result["annotated_image"])
            print(f"✓ Annotated image saved: {args.output}")

    elif args.batch:
        if not os.path.isdir(args.batch):
            print(f"❌ Not a directory: {args.batch}")
            sys.exit(1)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [
            f for f in os.listdir(args.batch)
            if os.path.splitext(f)[1].lower() in exts
        ]

        if not images:
            print(f"No images found in {args.batch}")
            sys.exit(0)

        print(f"\nBatch processing {len(images)} images from {args.batch}\n")

        for fname in sorted(images):
            path = os.path.join(args.batch, fname)
            try:
                result = predictor.analyse(path)
                logger.log(result, source=path)
                print_result(result)
            except Exception as e:
                print(f"  ❌ {fname}: {e}")

    if args.csv:
        logger.export_csv(args.csv)
        print(f"✓ CSV exported: {args.csv}")


if __name__ == "__main__":
    main()
