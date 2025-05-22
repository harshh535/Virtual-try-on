import os
import sys
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
import subprocess

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
    return

def clear_folder(folder_path):
    """
    Deletes everything inside `folder_path` and recreates it empty.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    print(f"🗑️ Cleared {folder_path}")

def update_test_pairs(image_folder, test_pairs_file, cloth_name):
    """
    Overwrites test_pairs.txt with lines:
        <model_image> <cloth_name>
    for every model in datasets/test/image.
    """
    if not os.path.exists(image_folder):
        print(f"❌ ERROR: Model image folder missing → {image_folder}")
        return

    model_images = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))]
    if not model_images:
        print(f"⚠️ WARNING: No model images found in → {image_folder}")

    with open(test_pairs_file, "w") as fp:
        for model_fn in model_images:
            fp.write(f"{model_fn} {cloth_name}\n")
    print(f"✅ test_pairs.txt updated → {test_pairs_file}  (paired '{cloth_name}' with {len(model_images)} models)")

def main(cloth_path):
    # ─── Determine directories relative to this script ───
    base_dir = os.path.dirname(os.path.abspath(__file__))

    image_folder      = os.path.join(base_dir, "datasets", "test", "image")
    cloth_mask_folder = os.path.join(base_dir, "datasets", "test", "cloth-mask")
    test_pairs_file   = os.path.join(base_dir, "datasets", "test", "test_pairs.txt")
    results_folder    = os.path.join(base_dir, "results")

    # 1) Ensure required folders exist / clear old outputs
    os.makedirs(cloth_mask_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    clear_folder(results_folder)

    # 2) Check that the model image folder is there
    if not os.path.exists(image_folder):
        print(f"❌ ERROR: Cannot find the models folder → {image_folder}")
        return

    # 3) Build the cloth-mask filename
    cloth_name      = Path(cloth_path).name
    cloth_mask_path = os.path.join(cloth_mask_folder, cloth_name)

    # 4) Generate the binary mask for this cloth
    generate_cloth_mask(cloth_path, cloth_mask_path)

    # 5) Debug: print what’s in the image folder
    print(f"📂 Checking model-images folder: {image_folder}")
    print(f"✅ Exists? {os.path.exists(image_folder)}")
    print(f"📜 Contents: {os.listdir(image_folder)}")

    # 6) Update test_pairs.txt (paired to every model)
    update_test_pairs(image_folder, test_pairs_file, cloth_name)

    # 7) Run test.py (virtual try-on) using same Python interpreter
    test_py_path = os.path.join(base_dir, "test.py")
    cmd = [
        sys.executable,
        test_py_path,
        "--name", "virtual_tryon",
        "--dataset_dir", os.path.join(base_dir, "datasets"),
        "--dataset_list", "test/test_pairs.txt",
        "--save_dir", os.path.join(base_dir, "results")
    ]
    print("🚀 Running test.py with:", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(f"⚠️ stderr from test.py:\n{proc.stderr}")

    if proc.returncode != 0:
        print("❌ test.py failed!")
        return

    # 8) Final report
    print("✅ Virtual try-on pipeline complete. Check → results/ for output images.")
    files = [f for f in os.listdir(results_folder) if os.path.isfile(os.path.join(results_folder, f))]
    if files:
        print(f"🎉 Found {len(files)} result file(s) in 'results/':")
        for f in files:
            print("   -", f)
    else:
        print("⚠️ 'results/' folder is empty! No output files found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate cloth-mask → test_pairs.txt → test.py")
    parser.add_argument(
        "cloth_path",
        type=str,
        help="Path to the cloth image (e.g. `/home/user/project/datasets/test/cloth/myshirt.jpg`)."
    )
    args = parser.parse_args()
    main(args.cloth_path)
