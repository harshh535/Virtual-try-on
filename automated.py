# automated.py

import os
import sys
import argparse
import shutil

import cv2
import numpy as np
from pathlib import Path
import subprocess

def generate_cloth_mask(input_path: str, output_path: str):
    """
    Generates a binary cloth mask for a given clothing image and saves it.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: Cloth image not found → {input_path}")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Error: Unable to read image → {input_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding (invert so cloth becomes white on black)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing (15×15 kernel)
    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask (same size as gray)
    mask = np.zeros_like(gray, dtype=np.uint8)

    # Keep only contours with area > min_area
    min_area = 5000
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Write mask to disk
    cv2.imwrite(output_path, mask)
    print(f"✅ Cloth mask saved at → {output_path}")
    return


def clear_results_folder(results_folder: str):
    """
    Deletes everything inside results_folder so each run starts fresh.
    """
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder, exist_ok=True)
    print("🗑️ Cleared results/ folder")


def update_test_pairs(image_folder: str, test_pairs_file: str, cloth_name: str):
    """
    Overwrites test_pairs.txt with lines:
        <model_image> <cloth_name>
    for every model in datasets/test/image.
    """
    if not os.path.exists(image_folder):
        print(f"❌ ERROR: Image folder not found → {image_folder}")
        return

    model_images = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".png"))
    ]
    if not model_images:
        print(f"⚠️ WARNING: No model images found in → {image_folder}")

    with open(test_pairs_file, "w") as fp:
        for model_fn in model_images:
            fp.write(f"{model_fn} {cloth_name}\n")

    print(f"✅ test_pairs.txt updated → {test_pairs_file}  (paired '{cloth_name}' with {len(model_images)} models)")


def main(cloth_path: str):
    """
    1. Clears ./results/
    2. Generates cloth-mask at datasets/test/cloth-mask/<cloth_name>
    3. Rewrites datasets/test/test_pairs.txt so <cloth_name> is paired to every file in datasets/test/image/
    4. Calls `test.py` with the appropriate flags.
    5. Reports which files ended up in ./results/.
    """
    base_dir       = os.path.dirname(os.path.abspath(__file__))
    image_folder   = os.path.join(base_dir, "datasets", "test", "image")
    cloth_mask_dir = os.path.join(base_dir, "datasets", "test", "cloth-mask")
    test_pairs_txt = os.path.join(base_dir, "datasets", "test", "test_pairs.txt")
    results_folder = os.path.join(base_dir, "results")

    # 1) Clear any old outputs in results/
    clear_results_folder(results_folder)

    # 2) Ensure required folders exist
    os.makedirs(cloth_mask_dir, exist_ok=True)

    # 3) Verify that the model image folder exists
    if not os.path.exists(image_folder):
        print(f"❌ ERROR: Model images folder missing → {image_folder}")
        sys.exit(1)

    # 4) Prepare cloth paths
    cloth_name      = Path(cloth_path).name
    cloth_mask_path = os.path.join(cloth_mask_dir, cloth_name)

    # 5) Generate cloth mask
    generate_cloth_mask(cloth_path, cloth_mask_path)

    # 6) Update test_pairs.txt inside datasets/test/
    update_test_pairs(image_folder, test_pairs_txt, cloth_name)

    # 7) Run test.py (virtual try-on)
    test_py_path = os.path.join(base_dir, "test.py")
    cmd = [
        sys.executable,
        test_py_path,
        "--name", "virtual_tryon",
        "--dataset_dir", os.path.join(base_dir, "datasets"),
        "--dataset_list", "test_pairs.txt",
        "--save_dir", os.path.join(base_dir, "results")
    ]
    print("\n🚀 Running test.py with:", " ".join(cmd), "\n")

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
        sys.exit(1)

    print("✅ Virtual try-on pipeline complete. Check → results/ for output images.\n")

    # 8) Report files in results/
    if os.path.exists(results_folder):
        files = [
            f for f in os.listdir(results_folder)
            if os.path.isfile(os.path.join(results_folder, f))
        ]
        if files:
            print(f"🎉 Found {len(files)} file(s) in 'results/':")
            for f in files:
                print("   -", f)
        else:
            print("⚠️ 'results/' folder is empty! No output files found.")
    else:
        print("⚠️ 'results/' folder does not exist!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate cloth-mask → test_pairs.txt → test.py"
    )
    parser.add_argument(
        "cloth_path",
        type=str,
        help="Path to the cloth image (e.g. ./datasets/test/cloth/myshirt.jpg)."
    )
    args = parser.parse_args()
    main(args.cloth_path)
