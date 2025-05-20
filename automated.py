# automated.py
# automated.py

import os
import sys
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
import subprocess

import sys, subprocess
print("▶︎ PYTHON VERSION:", sys.version)
print("▶︎ pip freeze:")
subprocess.run([sys.executable, "-m", "pip", "freeze"])


def generate_cloth_mask(input_path, output_path):
    """
    Generates a binary cloth mask for a given clothing image.
    Saves the mask (PNG) at output_path.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: File not found → {input_path}")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Error: Unable to read the image → {input_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding (invert so cloth becomes white on black)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing (15×15 kernel)
    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask (same size as gray)
    mask = np.zeros_like(gray)

    # Keep only contours with area > min_area
    min_area = 5000
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Write mask to disk
    cv2.imwrite(output_path, mask)
    print(f"✅ Cloth mask saved at → {output_path}")
    return output_path


def clear_results_folder(results_folder):
    """
    Deletes everything inside results_folder (files & subfolders),
    so each run starts with an empty ./results/ directory.
    """
    if os.path.exists(results_folder):
        for entry in os.listdir(results_folder):
            entry_path = os.path.join(results_folder, entry)
            try:
                if os.path.isfile(entry_path) or os.path.islink(entry_path):
                    os.unlink(entry_path)
                elif os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
            except Exception as e:
                print(f"❌ Failed to delete {entry_path}: {e}")
        print("🗑️ Cleared previous results from → results/")
    else:
        # If it doesn’t exist, create it
        os.makedirs(results_folder, exist_ok=True)
        print("🗑️ Created fresh results/ folder.")


def update_test_pairs(image_folder, test_pairs_file, cloth_name):
    """
    Overwrites test_pairs.txt with lines:
        <model_image> <cloth_name>
    for every model in datasets/test/image.
    """
    if not os.path.exists(image_folder):
        print(f"❌ ERROR: image_folder is missing → {image_folder}")
        return

    model_images = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))]
    if not model_images:
        print("⚠️ WARNING: No model images found in →", image_folder)

    # Write one line per model
    with open(test_pairs_file, "w") as fp:
        for model_fn in model_images:
            fp.write(f"{model_fn} {cloth_name}\n")

    print(f"✅ test_pairs.txt updated → {test_pairs_file}  (paired '{cloth_name}' with {len(model_images)} models)")


def main(cloth_path):
    """
    1. Clears ./results/
    2. Generates cloth-mask at datasets/test/cloth-mask/<cloth_name>
    3. Rewrites datasets/test/test_pairs.txt so <cloth_name> is paired to every file in datasets/test/image/
    4. Calls `test.py --name virtual_tryon --dataset_dir ./datasets --dataset_list test_pairs.txt --save_dir ./results`
       using the same interpreter (sys.executable).
    5. Leaves all outputs in ./results/, which shopkeeper_dashboard.py can then read and upload to Firebase.
    """
    # ─── Determine directories relative to this script ───
    base_dir = os.path.dirname(os.path.abspath(__file__))

    image_folder       = os.path.join(base_dir, "datasets", "test", "image")
    cloth_mask_folder  = os.path.join(base_dir, "datasets", "test", "cloth-mask")
    # NOTE: Write test_pairs.txt inside datasets/test/, not datasets/
    test_pairs_file    = os.path.join(base_dir, "datasets", "test", "test_pairs.txt")
    results_folder     = os.path.join(base_dir, "results")

    # 1) Ensure required folders exist
    os.makedirs(cloth_mask_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # 2) Clear any old outputs in results/
    clear_results_folder(results_folder)

    # 3) Check that the model image folder is there
    if not os.path.exists(image_folder):
        print(f"❌ ERROR: Cannot find the models folder → {image_folder}")
        return

    # 4) Build the cloth-mask filename
    cloth_name       = Path(cloth_path).name
    cloth_mask_path  = os.path.join(cloth_mask_folder, cloth_name)

    # 5) Generate the binary mask for this cloth
    generate_cloth_mask(cloth_path, cloth_mask_path)

    # 6) Update test_pairs.txt (inside datasets/test/) so that every model in image_folder is paired with cloth_name
    update_test_pairs(image_folder, test_pairs_file, cloth_name)

    # 7) Run test.py (virtual try-on) with the same Python interpreter and correct flags
    test_py_path = os.path.join(base_dir, "test.py")
    cmd = [
        sys.executable,
        test_py_path,
        "--name", "virtual_tryon",
        "--dataset_dir", os.path.join(base_dir, "datasets"),
        "--dataset_list", "test_pairs.txt",
        "--save_dir",    os.path.join(base_dir, "results")
    ]
    print("🚀 Running test.py with:", " ".join(cmd))

    try:
        completed = subprocess.run(
            cmd,
            cwd=base_dir,
            check=True,
            capture_output=True,
            text=True
        )
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print("⚠️ stderr from test.py:\n", completed.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Virtual try-on (`test.py`) failed with exit code {e.returncode}")
        print("stdout:\n", e.stdout)
        print("stderr:\n", e.stderr)
        return

    print("✅ Virtual try-on pipeline complete. Check → results/ for output images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate cloth-mask → test_pairs.txt → test.py")
    parser.add_argument(
        "cloth_path",
        type=str,
        help="Path to the cloth image (e.g. `/tmp/uploaded_shirt.png`)."
    )
    args = parser.parse_args()
    main(args.cloth_path)
