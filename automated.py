import os
import sys
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
import subprocess
import json

def generate_cloth_mask(input_path, output_path):
    """
    Generates a binary cloth mask for a given clothing image.
    Saves the mask (PNG) at output_path.
    """
    print("STEP 5: Generating cloth mask")
    if not os.path.exists(input_path):
        print(f"❌ Error: File not found → {input_path}")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ Error: Unable to read the image → {input_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)

    min_area = 5000
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    cv2.imwrite(output_path, mask)
    print(f"✅ Cloth mask saved at → {output_path}")

def clear_folder(folder_path):
    """
    Deletes and recreates `folder_path` so it’s empty.
    """
    print(f"STEP 2: Clearing folder → {folder_path}")
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    print(f"🗑️ Cleared {folder_path}")

def update_test_pairs(image_folder, test_pairs_file, cloth_name):
    """
    Overwrites test_pairs.txt so that every model in `image_folder` is paired with `cloth_name`.
    """
    print("STEP 7: Updating test_pairs.txt")
    if not os.path.exists(image_folder):
        print(f"❌ ERROR: Model-image folder missing → {image_folder}")
        return

    model_images = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))]
    print(f"📦 Found {len(model_images)} model images in {image_folder} → {model_images}")
    if not model_images:
        print(f"⚠️ WARNING: No model images to pair with {cloth_name}.")

    with open(test_pairs_file, "w") as fp:
        for model_fn in model_images:
            fp.write(f"{model_fn} {cloth_name}\n")
    print(f"✅ test_pairs.txt updated → {test_pairs_file} (paired '{cloth_name}' with {len(model_images)} models)")

def main(cloth_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("STEP 1: Starting automated pipeline")
    print(f"→ Base directory: {base_dir}")
    print(f"→ Received cloth_path: {cloth_path}")

    # 1) Folders & paths
    image_folder      = os.path.join(base_dir, "datasets", "test", "image")
    cloth_folder      = os.path.join(base_dir, "datasets", "test", "cloth")
    cloth_mask_folder = os.path.join(base_dir, "datasets", "test", "cloth-mask")
    test_pairs_file   = os.path.join(base_dir, "datasets", "test", "test_pairs.txt")
    results_folder    = os.path.join(base_dir, "results")

    print("STEP 1.1: Ensuring required directories exist")
    os.makedirs(cloth_folder, exist_ok=True)
    os.makedirs(cloth_mask_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    print(f"→ Created/Verified:\n   • {cloth_folder}\n   • {cloth_mask_folder}\n   • {results_folder}")

    # 2) Clear out any previous results
    clear_folder(results_folder)

    # 3) Check for model-image folder
    print("STEP 3: Checking for model-image folder")
    if not os.path.exists(image_folder):
        print(f"❌ ERROR: Cannot find model-image folder → {image_folder}")
        return
    print(f"✅ Found model-image folder → {image_folder}")

    # 4) Copy uploaded cloth into datasets/test/cloth/, only if it’s not already there
    print("STEP 4: Copying uploaded cloth image")
    cloth_name      = Path(cloth_path).name
    cloth_dest_path = os.path.join(cloth_folder, cloth_name)
    if os.path.abspath(cloth_path) != os.path.abspath(cloth_dest_path):
        shutil.copy(cloth_path, cloth_dest_path)
        print(f"📥 Copied cloth to → {cloth_dest_path}")
    else:
        print(f"↩️ Source and destination are the same ({cloth_dest_path}), skipping copy.")

    # 5) Generate cloth mask
    cloth_mask_path = os.path.join(cloth_mask_folder, cloth_name)
    generate_cloth_mask(cloth_dest_path, cloth_mask_path)

    # 6) Debug: show contents of image_folder
    print("STEP 6: Listing contents of image-folder")
    print(f"📂 Checking image-folder: {image_folder}")
    print(f"✅ Exists? {os.path.exists(image_folder)}")
    print(f"📜 Contents: {os.listdir(image_folder)}")

    # 7) Update test_pairs.txt
    update_test_pairs(image_folder, test_pairs_file, cloth_name)

    # 8) Print test_pairs.txt contents for debugging
    print("STEP 8: Printing test_pairs.txt contents for verification")
    with open(test_pairs_file, "r") as f:
        for line in f:
            print("   " + line.strip())

    # 9) Run test.py
    print("STEP 9: Running test.py inference")
    test_py_path = os.path.join(base_dir, "test.py")
    cmd = [
        sys.executable,
        test_py_path,
        "--name", "virtual_tryon",
        "--dataset_dir", "datasets",
        "--dataset_list", "test/test_pairs.txt",
        "--save_dir", "results"
    ]
    print("🚀 Executing test.py with:", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if proc.stdout:
        print(f"--- test.py STDOUT ---\n{proc.stdout}")
    if proc.stderr:
        print(f"⚠️ stderr from test.py:\n{proc.stderr}")

    if proc.returncode != 0:
        print("❌ test.py failed with exit code", proc.returncode)
        return
    print("✅ test.py completed successfully")

    # 10) List whatever ended up in results/
    print("STEP 10: Verifying results/")
    saved = []
    for fname in os.listdir(results_folder):
        if fname.lower().endswith((".jpg", ".png")):
            saved.append(fname)
    if saved:
        print(f"🎉 Found {len(saved)} output file(s) in results/:")
        for f in saved:
            print("   -", f)
    else:
        print("⚠️ 'results/' folder is empty! No output files found.")

    print("STEP 11: Automated pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate: generate cloth-mask → update test_pairs.txt → run test.py"
    )
    parser.add_argument(
        "cloth_path",
        type=str,
        help="Local path to the cloth image (e.g. `/mnt/data/cloth123.jpg`)."
    )
    args = parser.parse_args()
    main(args.cloth_path)
    parser.add_argument(
        "cloth_path",
        type=str,
        help="Local path to the cloth image (e.g. `/mnt/data/cloth123.jpg`)."
    )
    args = parser.parse_args()
    main(args.cloth_path)
