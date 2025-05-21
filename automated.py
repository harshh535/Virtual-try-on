import os
import sys
import shutil
import cv2
import argparse
import subprocess
import numpy as np
from pathlib import Path

def clear_results_folder(results_folder):
    """
    Deletes everything inside results_folder so each run starts fresh.
    """
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder, exist_ok=True)


def generate_cloth_mask(input_path, output_path):
    """
    Generates a binary cloth mask for a given clothing image.
    Saves the mask (PNG) at output_path.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cloth image not found: {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        raise RuntimeError(f"Unable to read the cloth image → {input_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding (invert so cloth becomes white on black)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing (15×15 kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a proper single‐channel (uint8) mask of the same size as gray
    mask = np.zeros_like(gray, dtype=np.uint8)

    # Keep only contours with area > min_area
    min_area = 5000
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Ensure the output folder exists, then write mask to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)


def update_test_pairs(image_folder, test_pairs_file, cloth_name):
    """
    Overwrites test_pairs.txt so every model in image_folder pairs with cloth_name.
    """
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Model images folder not found: {image_folder}")

    model_images = [fn for fn in os.listdir(image_folder) if fn.lower().endswith((".jpg", ".png"))]
    if not model_images:
        raise RuntimeError(f"No model images found in → {image_folder}")

    os.makedirs(os.path.dirname(test_pairs_file), exist_ok=True)
    with open(test_pairs_file, "w") as fp:
        for model_fn in model_images:
            fp.write(f"{model_fn} {cloth_name}\n")


def main():
    parser = argparse.ArgumentParser(description="Automate cloth-mask → test_pairs.txt → test.py")
    parser.add_argument(
        "cloth_path",
        type=str,
        help="Path to the uploaded cloth image (e.g. './datasets/test/cloth/my_shirt.jpg')"
    )
    args = parser.parse_args()

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    cloth_path = args.cloth_path
    cloth_name = Path(cloth_path).name

    # Directories (relative to repo root)
    image_folder      = os.path.join(base_dir, "datasets", "test", "image")
    cloth_mask_folder = os.path.join(base_dir, "datasets", "test", "cloth-mask")
    test_pairs_file   = os.path.join(base_dir, "datasets", "test", "test_pairs.txt")
    results_folder    = os.path.join(base_dir, "results")

    # 1) Clear any previous outputs
    clear_results_folder(results_folder)

    # 2) Generate binary mask for the new cloth
    os.makedirs(cloth_mask_folder, exist_ok=True)
    mask_path = os.path.join(cloth_mask_folder, cloth_name)
    generate_cloth_mask(cloth_path, mask_path)

    # 3) Update test_pairs.txt so each model in datasets/test/image is paired with this cloth
    update_test_pairs(image_folder, test_pairs_file, cloth_name)

    # 4) Run test.py (CPU‐only) with the same Python interpreter
    test_py = os.path.join(base_dir, "test.py")
    cmd = [
        sys.executable,
        test_py,
        "--name", "virtual_tryon",
        "--dataset_dir", os.path.join(base_dir, "datasets"),
        "--dataset_list", "test_pairs.txt",
        "--save_dir", results_folder
    ]

    completed = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True)
    if completed.returncode != 0:
        print("❌ test.py failed. stderr:\n", completed.stderr)
        sys.exit(1)

    # 5) Report final contents of results/
    if os.path.exists(results_folder):
        saved = [fn for fn in os.listdir(results_folder) if os.path.isfile(os.path.join(results_folder, fn))]
        if saved:
            print(f"🎉 Inference complete. Found {len(saved)} files in 'results/':")
            for fn in saved:
                print("   -", fn)
        else:
            print("⚠️ 'results/' folder is empty—no images generated.")
    else:
        print("⚠️ 'results/' folder does not exist!")


if __name__ == "__main__":
    main()
