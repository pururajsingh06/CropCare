from pathlib import Path


DATASET_PATH = Path(r"C:\Users\acer\OneDrive\Desktop\minor_project\minor_dateset")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

if not DATASET_PATH.exists():
    print(f"Dataset folder not found: {DATASET_PATH}")

else:
    total_images = 0
    total_classes = 0

    print("\nDATASET SUMMARY")
    print("-" * 70)

    for folder in sorted(DATASET_PATH.iterdir()):
        if folder.is_dir():
            images = [
                file
                for file in folder.rglob("*")
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            ]

            count = len(images)

            total_classes += 1
            total_images += count

            print(f"{folder.name:<45} {count:>7} images")

    print("-" * 70)
    print(f"Total classes: {total_classes}")
    print(f"Total images:  {total_images}")
