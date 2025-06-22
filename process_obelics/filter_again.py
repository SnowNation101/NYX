import os
import json
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

with open("process_obelics/generated_qa.json", "r") as f:
    generated_qa = json.load(f)

image_dir = "/fs/archive/share/mm_datasets/obelics_images"

processed_qa = []
error_count = 0
skipped_count = 0

for item in generated_qa:
    images = item.get("images", [])
    corrupted = False
    for image_name in images:
        try:
            image_path = os.path.join(image_dir, image_name)
            with Image.open(image_path) as img:
                img.verify()  # Check format
            with Image.open(image_path) as img:
                img.load()   # Check decoding (will raise if truncated)
        except Exception as e:
            print(f"Error with image {image_name}: {e}")
            error_count += 1
            corrupted = True
            break  # Skip the whole item
    if not corrupted:
        processed_qa.append(item)
    else:
        skipped_count += 1

with open("process_obelics/processed_qa.json", "w") as f:
    json.dump(processed_qa, f, ensure_ascii=True, indent=2)

print(f"Total corrupted/truncated images: {error_count}")
print(f"Total skipped items: {skipped_count}")
print(f"Total remaining valid items: {len(processed_qa)}")
