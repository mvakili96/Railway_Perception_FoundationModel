import argparse
import copy
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

GENERIC_PROMPTS = [
    "Which track bed corresponds to the ego-route based on rail continuity?",
    "Identify the track bed of the ego-route in this image by following the continuous rails.",
    "Which path remains continuous for the ego-route in this scene?",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-image validation JSON files for cropped RailSem19 "
            "images using a reason-segmentation template and cropped rail "
            "annotations."
        )
    )
    parser.add_argument(
        "--image-dir",
        default="dataset/reason_seg/ReasonSegRail/val",
        help="Directory containing the cropped validation images.",
    )
    parser.add_argument(
        "--egopath-json",
        default="dataset/metadata/rs19_validation_egopath_1024.json",
        help="Cropped validation ego-path annotation JSON.",
    )
    parser.add_argument(
        "--template-json",
        default="scripts/data/templates/reason_seg_validation_template.json",
        help="Template JSON whose reason-segmentation structure will be copied.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of cropped images to process from the sorted file list.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_image_files(image_dir):
    image_dir = Path(image_dir)
    return sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_points(annotation):
    right_rail = annotation["right_rail"]
    left_rail = annotation["left_rail"]
    return right_rail + list(reversed(left_rail))


def generate_json_data(template, image_name, annotation):
    sample = copy.deepcopy(template)
    sample["text"] = list(GENERIC_PROMPTS)
    sample["shapes"][0]["image_name"] = image_name
    sample["shapes"][0]["points"] = build_points(annotation)
    return sample


def main():
    args = parse_args()

    template = load_json(args.template_json)
    egopath_data = load_json(args.egopath_json)

    if "shapes" not in template or not template["shapes"]:
        raise ValueError("Template JSON must contain at least one shape.")

    image_files = list_image_files(args.image_dir)[: args.limit]
    image_dir = Path(args.image_dir)

    written = 0
    for image_name in image_files:
        if image_name not in egopath_data:
            raise KeyError(f"{image_name} not found in {args.egopath_json}")

        output_data = generate_json_data(template, image_name, egopath_data[image_name])
        output_path = image_dir / f"{Path(image_name).stem}.json"

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=2)
            handle.write("\n")

        written += 1

    print(f"Wrote {written} JSON files to: {image_dir}")
    if image_files:
        print(f"First image processed: {image_files[0]}")
        print(f"Last image processed: {image_files[-1]}")


if __name__ == "__main__":
    main()
