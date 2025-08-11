import cv2
from pathlib import Path
from itertools import chain

def resize_and_pad(img, size=224):
    """Resize image while maintaining aspect ratio, and pad to desired size."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))

    delta_w = size - new_w
    delta_h = size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # black padding
    )
    return padded

def process_images(input_dir, output_dir, target_size=224):
    """Resize and pad images in all subfolders of input_dir, saving to output_dir."""
    total_found = total_resized = 0
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for subfolder in sorted(input_dir.iterdir()):
        if subfolder.is_dir():
            for img_path in subfolder.rglob("*"):
                if img_path.suffix.lower() in valid_exts:
                    total_found += 1
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"⚠️ Skipping unreadable image: {img_path}")
                        continue

                    resized = resize_and_pad(img, size=target_size)
                    save_path = output_dir / subfolder.name / img_path.name
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    if cv2.imwrite(str(save_path), resized):
                        total_resized += 1
                    else:
                        print(f"❌ Failed to save image: {save_path}")

    return total_found, total_resized

# Set paths
training_input = Path("C:/Users/tomla/Documents/Projects/brain_tumor_classifier/data/datasets/sartajbhuvaji/brain-tumor-classification-mri/versions/2/Training")
testing_input  = Path("C:/Users/tomla/Documents/Projects/brain_tumor_classifier/data/datasets/sartajbhuvaji/brain-tumor-classification-mri/versions/2/Testing")

training_output = Path("C:/Users/tomla/Documents/Projects/brain_tumor_classifier/data/training")
testing_output  = Path("C:/Users/tomla/Documents/Projects/brain_tumor_classifier/data/testing")

# Run processing
print("📂 Processing training images...")
train_found, train_resized = process_images(training_input, training_output)

print(f"\n✅ Total files found: {train_found}")
print(f"✅ Total files resized and saved: {train_resized}")

print("📂 Processing testing images...")
test_found, test_resized = process_images(testing_input, testing_output)

print(f"\n✅ Total files found: {test_found}")
print(f"✅ Total files resized and saved: {test_resized}")
