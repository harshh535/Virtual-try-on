import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import gdown

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior")

# ─── Google Drive IDs ─────────────────────────────────────────────────────────
SEG_CKPT_ID   = "1Hb_y7M4pQlrKh6m4-2Mo_m_KU1IcT6DB"
GMM_CKPT_ID   = "1gtagvr1I8Dq4ejnpQ51fZ9G9sCloKgyh"
ALIAS_CKPT_ID = "1vWoDdaiWF0Zuv8Md9bn7q-xLUUIjXReY"

CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR    = "./results"

def download_if_not_exists(file_id, dest_path):
    """
    If checkpoint `dest_path` is missing, download from Google Drive.
    Otherwise, report that it’s already there.
    """
    print("▶️ [download_if_not_exists] Checking for checkpoint:", dest_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print("   ☑️  Already exists:", dest_path)
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print("   ⬇️  Downloading", os.path.basename(dest_path), "from Google Drive...")
    gdown.download(url, dest_path, quiet=False)
    print("   ✅  Download complete →", dest_path)

def get_opt():
    """
    Builds an argparse.Namespace containing all options needed by load_models()
    and run_inference().
    """
    print("▶️ [get_opt] Parsing CLI arguments")
    parser = argparse.ArgumentParser(description="Test Virtual Try-On")

    # ─── Model / network initialization arguments ──────────────────────────────
    parser.add_argument(
        "--init_type",
        type=str,
        default="normal",
        help="network weight initialization (default: normal)"
    )
    parser.add_argument(
        "--init_variance",
        type=float,
        default=0.02,
        help="initialization variance (default: 0.02)"
    )
    parser.add_argument("--name",            type=str, required=True)
    parser.add_argument("--dataset_dir",     type=str, default="./datasets")
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="test",
        help="which subfolder under dataset_dir to use"
    )
    parser.add_argument(
        "--dataset_list",
        type=str,
        default="test/test_pairs.txt",
        help="relative path (inside dataset_dir) listing cloth-model pairs"
    )
    parser.add_argument("--save_dir",        type=str, default="./results")
    parser.add_argument("--checkpoint_dir",  type=str, default="./checkpoints")
    parser.add_argument("--load_height",     type=int, default=1024)
    parser.add_argument("--load_width",      type=int, default=768)
    parser.add_argument("--semantic_nc",     type=int, default=13)
    parser.add_argument("--grid_size",       type=int, default=5)
    parser.add_argument("--norm_G",          type=str, default="spectralaliasinstance")
    parser.add_argument("--ngf",             type=int, default=64)
    parser.add_argument(
        "--num_upsampling_layers",
        choices=["normal", "more", "most"],
        default="most",
        help="how many upsampling layers in the network"
    )
    parser.add_argument("--display_freq",    type=int, default=1)
    # ──────────────────────────────────────────────────────────────────────────

    # ─── DataLoader arguments ─────────────────────────────────────────────────
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="whether to shuffle the dataset (default: False)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for DataLoader (default: 1)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of worker threads for DataLoader (default: 0)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="alias for --num_workers; used by VITONDataLoader (default: 0)"
    )
    # ──────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()
    print("▶️ [get_opt] Parsed arguments:", args)
    return args

def load_models(opt):
    """
    1. Ensures each checkpoint (SEG, GMM, ALIAS) is present (download if needed).
    2. Constructs SegGenerator, GMM, ALIASGenerator.
    3. Loads their .pth weights, sets `.eval()`, and returns them.
    """
    print("▶️ [load_models] Creating directories:", opt.checkpoint_dir, "/", opt.save_dir)
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)

    seg_path   = os.path.join(opt.checkpoint_dir, "seg_final.pth")
    gmm_path   = os.path.join(opt.checkpoint_dir, "gmm_final.pth")
    alias_path = os.path.join(opt.checkpoint_dir, "alias_final.pth")

    print("▶️ [load_models] Looking for SEG checkpoint at:", seg_path)
    download_if_not_exists(SEG_CKPT_ID, seg_path)

    print("▶️ [load_models] Looking for GMM checkpoint at:", gmm_path)
    download_if_not_exists(GMM_CKPT_ID, gmm_path)

    print("▶️ [load_models] Looking for ALIAS checkpoint at:", alias_path)
    download_if_not_exists(ALIAS_CKPT_ID, alias_path)

    # Construct networks
    print("▶️ [load_models] Constructing SegGenerator")
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)

    print("▶️ [load_models] Constructing GMM")
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)

    prev_sem = opt.semantic_nc
    opt.semantic_nc = 7
    print("▶️ [load_models] Constructing ALIASGenerator")
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = prev_sem

    # Load weights
    print("▶️ [load_models] Loading SEG weights...")
    load_checkpoint(seg, seg_path)
    print("   ☑️  SEG weights loaded.")

    print("▶️ [load_models] Loading GMM weights...")
    load_checkpoint(gmm, gmm_path)
    print("   ☑️  GMM weights loaded.")

    print("▶️ [load_models] Loading ALIAS weights...")
    load_checkpoint(alias, alias_path)
    print("   ☑️  ALIAS weights loaded.")

    seg.eval()
    gmm.eval()
    alias.eval()

    print("▶️ [load_models] Networks ready. Returning `seg`, `gmm`, `alias`.")
    return seg, gmm, alias

def run_inference(opt, seg, gmm, alias):
    """
    1️⃣ Reads and prints `test_pairs.txt`.
    2️⃣ Builds `VITONDataset` & `VITONDataLoader`.
    3️⃣ For each batch: predicts segmentation → warps cloth → aliases final.
    4️⃣ Saves each output under `opt.save_dir/{modelName}_{clothName}.jpg`
    5️⃣ Prints debugging info at every major step.
    """

    # ─── Fix shared‐memory issue by using file_system strategy ──────────────────
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
    print("▶️ [run_inference] Set multiprocessing sharing strategy to 'file_system'")

    up    = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))

    print("▶️ [run_inference] Reading test-pairs from:", os.path.join(opt.dataset_dir, opt.dataset_list))
    pairs_path = os.path.join(opt.dataset_dir, opt.dataset_list)
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"❌ Cannot find {pairs_path}")
    with open(pairs_path, "r") as f:
        lines = f.read().strip().splitlines()
    print(f"   ▶️ {len(lines)} lines in test_pairs.txt:")
    for ln in lines:
        print(f"      {ln}")

    print("▶️ [run_inference] Building VITONDataset")
    test_dataset = VITONDataset(opt)
    print(f"   ▶️ Dataset length = {len(test_dataset)}")

    print("▶️ [run_inference] Building VITONDataLoader with num_workers =", opt.num_workers)
    test_loader  = VITONDataLoader(opt, test_dataset)
    loader_len = len(test_loader.data_loader)
    print(f"   ▶️ DataLoader length (batches) = {loader_len}")

    if len(test_dataset) == 0:
        raise ValueError("⚠️ No test pairs found! Exiting inference.")

    print("▶️ [run_inference] Starting batch loop")
    total_saved = 0
    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            print(f"\n   🌀 [Batch {i+1}/{loader_len}]")
            img_names      = inputs['img_name']
            c_names        = inputs['c_name']['unpaired']
            img_agnostic   = inputs['img_agnostic']
            parse_agnostic = inputs['parse_agnostic']
            pose           = inputs['pose']
            c              = inputs['cloth']['unpaired']
            cm             = inputs['cloth_mask']['unpaired']

            print(f"      • img_names    = {img_names}")
            print(f"      • cloth_names  = {c_names}")

            # ── Part 1: Segmentation prediction ───────────────────────────────
            print("      ▶️ Building segmentation input")
            seg_input = torch.cat([
                F.interpolate(cm,            (256, 192), mode='bilinear'),
                F.interpolate(c * cm,        (256, 192), mode='bilinear'),
                F.interpolate(parse_agnostic,(256, 192), mode='bilinear'),
                F.interpolate(pose,          (256, 192), mode='bilinear'),
                gen_noise((c.size(0), 1, 256, 192))
            ], dim=1)

            print("      ▶️ Running SegGenerator")
            parse_pred_down = seg(seg_input)
            print("      ▶️ Obtaining segmentation output")
            parse_pred      = gauss(up(parse_pred_down)).argmax(dim=1)[:, None]

            print("      ▶️ Converting to one-hot 13-channel")
            parse_old = torch.zeros(
                parse_pred.size(0), opt.semantic_nc,
                opt.load_height, opt.load_width
            ).to(parse_pred.device)
            parse_old.scatter_(1, parse_pred, 1.0)

            print("      ▶️ Building 7-channel parse")
            parse = torch.zeros(
                parse_pred.size(0), 7,
                opt.load_height, opt.load_width
            ).to(parse_pred.device)
            parse[:, 0] = parse_old[:, 0]
            parse[:, 1] = parse_old[:, [2, 4, 7, 8, 9, 10, 11]].sum(dim=1)
            parse[:, 2] = parse_old[:, 3]
            parse[:, 3] = parse_old[:, 1]
            parse[:, 4] = parse_old[:, 5]
            parse[:, 5] = parse_old[:, 6]
            parse[:, 6] = parse_old[:, 12]

            # ── Part 2: GMM warping ─────────────────────────────────────────
            print("      ▶️ Preparing inputs for GMM")
            agnostic_gmm     = F.interpolate(img_agnostic, (256, 192), mode='nearest')
            parse_cloth_gmm  = F.interpolate(parse[:, 2:3], (256, 192), mode='nearest')
            pose_gmm         = F.interpolate(pose, (256, 192), mode='nearest')
            c_gmm            = F.interpolate(c, (256, 192), mode='nearest')
            gmm_input        = torch.cat([parse_cloth_gmm, pose_gmm, agnostic_gmm], dim=1)

            print("      ▶️ Running GMM")
            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c       = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm      = F.grid_sample(cm, warped_grid, padding_mode='border')

            # ── Part 3: ALIAS final synthesis ───────────────────────────────
            print("      ▶️ Preparing inputs for ALIAS")
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask = torch.clamp(misalign_mask, min=0.0)
            parse_div     = torch.cat([parse, misalign_mask], dim=1)
            parse_div[:, 2:3] -= misalign_mask

            print("      ▶️ Running ALIASGenerator")
            output = alias(
                torch.cat([img_agnostic, pose, warped_c], dim=1),
                parse,
                parse_div,
                misalign_mask
            )

            print("      ▶️ Saving output images")
            save_names = []
            for j, img_name in enumerate(img_names):
                cloth_name = c_names[j]
                out_name   = f"{img_name.split('_')[0]}_{cloth_name}"
                save_names.append(out_name)

            save_images(output, save_names, opt.save_dir)
            print(f"      ✔️ Saved {len(save_names)} images: {save_names}")
            total_saved += len(save_names)

    # 4) Final tally
    print(f"\n🎉 Inference complete! Total images saved: {total_saved}")
    existing = [
        f for f in os.listdir(opt.save_dir)
        if f.lower().endswith((".jpg", ".png"))
    ]
    print("   • Currently in `", opt.save_dir, "` →", existing)
    print("▶️ Exiting run_inference()")

def main():
    print("▶️ Entered main()")
    opt = get_opt()
    print("▶️ Parsed CLI options")

    # Ensure save_dir exists
    os.makedirs(opt.save_dir, exist_ok=True)
    print("   • Created save_dir:", opt.save_dir)

    if not hasattr(opt, 'workers'):
        opt.workers = opt.num_workers
        print("   • Set opt.workers = opt.num_workers =", opt.workers)

    print("▶️ Calling load_models()")
    seg, gmm, alias = load_models(opt)

    print("▶️ Calling run_inference()")
    run_inference(opt, seg, gmm, alias)

    print("▶️ Exiting main()")

if __name__ == "__main__":
    main()
