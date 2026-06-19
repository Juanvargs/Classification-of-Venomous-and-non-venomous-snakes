import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
THUMB_SIZE = (180, 180)
LABEL_HEIGHT = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Crea laminas de inspeccion visual del dataset.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="reports/audits/dataset_audit/contact_sheets")
    parser.add_argument("--samples-per-group", type=int, default=16)
    return parser.parse_args()


def list_images(class_dir):
    return [
        path
        for path in Path(class_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def make_contact_sheet(paths, title, output_path):
    cols = 4
    rows = max(1, (len(paths) + cols - 1) // cols)
    width = cols * THUMB_SIZE[0]
    height = rows * (THUMB_SIZE[1] + LABEL_HEIGHT) + LABEL_HEIGHT
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")

    for index, path in enumerate(paths):
        row = index // cols
        col = index % cols
        x = col * THUMB_SIZE[0]
        y = LABEL_HEIGHT + row * (THUMB_SIZE[1] + LABEL_HEIGHT)

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail(THUMB_SIZE)

            thumb = Image.new("RGB", THUMB_SIZE, "white")
            offset = (
                (THUMB_SIZE[0] - image.width) // 2,
                (THUMB_SIZE[1] - image.height) // 2,
            )
            thumb.paste(image, offset)

        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + THUMB_SIZE[1] + 4), path.name[:24], fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main():
    args = parse_args()
    random.seed(SEED)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    for split_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            images = list_images(class_dir)
            sample_count = min(args.samples_per_group, len(images))
            selected = random.sample(images, sample_count)
            title = f"{split_dir.name} / {class_dir.name} ({len(images)} imagenes)"
            output_path = output_dir / f"{split_dir.name}_{class_dir.name.replace(' ', '_')}.jpg"
            make_contact_sheet(selected, title, output_path)
            print(output_path)


if __name__ == "__main__":
    main()
