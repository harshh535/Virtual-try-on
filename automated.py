import os
import cv2
import subprocess
import shutil

# Paths
CLOTH_DIR = "datasets/test/cloth"
CLOTH_MASK_DIR = "datasets/test/cloth-mask"
IMAGE_DIR = "datasets/test/image"
TEST_PAIRS_PATH = "datasets/test/test_pairs.txt"
RESULTS_DIR = "results"
CLOTH_NAME = "cloth.jpg"  # <-- change if using a different filename

def clear_results():
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("🧹 Cleared results/ folder")

def generate_cloth_mask():
    cloth_path = os.path.join(CLOTH_DIR, CLOTH_NAME)
    mask_path = os.path.join(CLOTH_MASK_DIR, CLOTH_NAME)

    if not os.path.exists(cloth_path):
        raise FileNotFoundError(f"❌ Cloth image not found: {cloth_path}")

    os.makedirs(CLOTH_MASK_DIR, exist_ok=True)
    cloth = cv2.imread(cloth_path)
    gray = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(mask_path, mask)
    print(f"🧵 Cloth mask generated at {mask_path}")

def update_test_pairs():
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"❌ Model images not found: {IMAGE_DIR}")

    model_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]
    if not model_files:
        raise ValueError("❌ No model images found in image/ directory")

    with open(TEST_PAIRS_PATH, 'w') as f:
        for model in model_files:
            f.write(f"{model} {CLOTH_NAME}\n")
    print(f"📄 test_pairs.txt updated with {len(model_files)} pairs")

def run_inference():
    print("🚀 Starting test.py inference...")
    subprocess.run([sys.executable, "test.py"], check=True)
    print("✅ Inference completed")

def main():
    clear_results()
    generate_cloth_mask()
    update_test_pairs()
    run_inference()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error during automation: {e}")
        exit(1)
