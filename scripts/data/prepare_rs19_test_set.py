import argparse
import json
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create 1024x1024 RS19 crops that are horizontally centered in the "
            "original image and bottom-aligned, then shift rail annotations into "
            "the cropped image coordinates."
        )
    )
    parser.add_argument(
        "--input-json",
        default="dataset/external/tepnet/egopath/rs19_egopath.json",
        help="Path to the original TEP-Net ego-path annotation JSON.",
    )
    parser.add_argument(
        "--image-dir",
        default="dataset/external/railsem19/test_images",
        help="Directory containing the held-out RailSem19 images.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/test/images",
        help="Directory where the cropped images will be written.",
    )
    parser.add_argument(
        "--output-json",
        default="dataset/test/rs19_egopath_1024.json",
        help="Path to the shifted annotation JSON for the new crops.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=1024,
        help="Square crop size in pixels.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_crop_bounds(width, height, crop_size):
    if width < crop_size or height < crop_size:
        raise ValueError(
            f"Image size {width}x{height} is smaller than crop size {crop_size}."
        )

    x0 = (width - crop_size) // 2
    x1 = x0 + crop_size
    y1 = height
    y0 = y1 - crop_size
    return x0, y0, x1, y1


def shift_visible_points(points, x0, y0, x1, y1):
    shifted = []
    for x, y in points:
        if x0 <= x < x1 and y0 <= y < y1:
            shifted.append([int(x - x0), int(y - y0)])
    return shifted


def build_cropped_annotation(annotation, x0, y0, x1, y1):
    return {
        "left_rail": shift_visible_points(annotation["left_rail"], x0, y0, x1, y1),
        "right_rail": shift_visible_points(annotation["right_rail"], x0, y0, x1, y1),
    }


def process_dataset(input_json, image_dir, output_dir, output_json, crop_size):
    annotations = load_json(input_json)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cropped_annotations = {}
    processed = 0

    for image_name, annotation in annotations.items():
        image_path = image_dir / image_name
        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        height, width = image.shape[:2]
        x0, y0, x1, y1 = compute_crop_bounds(width, height, crop_size)

        cropped_image = image[y0:y1, x0:x1]
        output_image_path = output_dir / image_name
        if not cv2.imwrite(str(output_image_path), cropped_image):
            raise ValueError(f"Failed to write cropped image: {output_image_path}")

        cropped_annotations[image_name] = build_cropped_annotation(
            annotation, x0, y0, x1, y1
        )
        processed += 1

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(cropped_annotations, handle)

    print(f"Processed images: {processed}")
    print(f"Cropped images saved to: {output_dir}")
    print(f"Cropped annotations saved to: {output_json}")


def main():
    args = parse_args()
    process_dataset(
        input_json=args.input_json,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        output_json=args.output_json,
        crop_size=args.crop_size,
    )


if __name__ == "__main__":
    main()
