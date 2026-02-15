#!/usr/bin/env python3
"""
Run combined inference on fruit images.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --image_dir path/to/images/
"""

import argparse
from pathlib import Path

import torch

from src.models.combined_classifier import CombinedFruitClassifier


def main():
    parser = argparse.ArgumentParser(description="Fruit classification inference")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to directory of images"
    )
    parser.add_argument(
        "--fruit_checkpoint",
        type=str,
        default="checkpoints/fruit_classifier/best_model.pth",
        help="Path to fruit classifier checkpoint"
    )
    parser.add_argument(
        "--freshness_checkpoint",
        type=str,
        default="checkpoints/freshness_classifier/best_model.pth",
        help="Path to freshness classifier checkpoint"
    )
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        print("Error: Please provide --image or --image_dir")
        return

    # Check checkpoints exist
    if not Path(args.fruit_checkpoint).exists():
        print(f"Error: Fruit checkpoint not found: {args.fruit_checkpoint}")
        print("Please train the fruit classifier first:")
        print("  python train_fruit_classifier.py --data_dir data/fruit_type")
        return

    if not Path(args.freshness_checkpoint).exists():
        print(f"Error: Freshness checkpoint not found: {args.freshness_checkpoint}")
        print("Please train the freshness classifier first:")
        print("  python train_freshness_classifier.py --data_dir data/freshness")
        return

    # Load class mappings
    fruit_classes_path = Path(args.fruit_checkpoint).parent / "classes.pth"
    freshness_classes_path = Path(args.freshness_checkpoint).parent / "classes.pth"

    if not fruit_classes_path.exists():
        print(f"Error: Fruit classes mapping not found: {fruit_classes_path}")
        return

    if not freshness_classes_path.exists():
        print(f"Error: Freshness classes mapping not found: {freshness_classes_path}")
        return

    fruit_meta = torch.load(fruit_classes_path, weights_only=True)
    freshness_meta = torch.load(freshness_classes_path, weights_only=True)

    # Initialize classifier
    print("Loading models...")
    classifier = CombinedFruitClassifier(
        fruit_checkpoint=args.fruit_checkpoint,
        freshness_checkpoint=args.freshness_checkpoint,
        fruit_classes=fruit_meta["classes"],
        freshness_classes=freshness_meta["classes"]
    )
    print("Models loaded!\n")

    if args.image:
        # Single image prediction
        if not Path(args.image).exists():
            print(f"Error: Image not found: {args.image}")
            return

        label, confidence = classifier.predict_single(args.image)

        print(f"Image: {args.image}")
        print(f"Prediction: {label}")
        print(f"  Fruit: {confidence['fruit']['label']} "
              f"({confidence['fruit']['confidence']:.1%})")
        print(f"  Freshness: {confidence['freshness']['label']} "
              f"({confidence['freshness']['confidence']:.1%})")

    elif args.image_dir:
        # Batch prediction
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"Error: Directory not found: {args.image_dir}")
            return

        # Find all images
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(ext))
            image_paths.extend(image_dir.glob(ext.upper()))

        if not image_paths:
            print(f"No images found in {args.image_dir}")
            return

        print(f"Processing {len(image_paths)} images...\n")

        results = classifier.predict_batch([str(p) for p in image_paths])

        for path, (label, confidence) in zip(image_paths, results):
            fruit_conf = confidence["fruit"]["confidence"]
            fresh_conf = confidence["freshness"]["confidence"]
            print(f"{path.name}: {label} "
                  f"(fruit: {fruit_conf:.0%}, freshness: {fresh_conf:.0%})")


if __name__ == "__main__":
    main()
