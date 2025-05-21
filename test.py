import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import gdown
from types import SimpleNamespace
from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images

# Google Drive checkpoint IDs
SEG_CKPT_ID = "1Hb_y7M4pQlrKh6m4-2Mo_m_KU1IcT6DB"
GMM_CKPT_ID = "1gtagvr1I8Dq4ejnpQ51fZ9G9sCloKgyh"
ALIAS_CKPT_ID = "1vWoDdaiWF0Zuv8Md9bn7q-xLUUIjXReY"

CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

def download_checkpoint(file_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {os.path.basename(dest_path)}...")
        gdown.download(url, dest_path, quiet=False)

def load_models():
    """Load all required models without Streamlit dependencies"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    seg_ckpt_path = os.path.join(CHECKPOINT_DIR, "seg_final.pth")
    gmm_ckpt_path = os.path.join(CHECKPOINT_DIR, "gmm_final.pth")
    alias_ckpt_path = os.path.join(CHECKPOINT_DIR, "alias_final.pth")

    # Download checkpoints if missing
    download_checkpoint(SEG_CKPT_ID, seg_ckpt_path)
    download_checkpoint(GMM_CKPT_ID, gmm_ckpt_path)
    download_checkpoint(ALIAS_CKPT_ID, alias_ckpt_path)

    # Configuration
    opt = SimpleNamespace(
        load_height=1024,
        load_width=768,
        semantic_nc=13,
        init_type='xavier',
        init_variance=0.02,
        norm_G='spectralaliasinstance',
        ngf=64,
        num_upsampling_layers='most',
        batch_size=1,
        workers=0,
        shuffle=False,
        dataset_dir=os.path.abspath("./datasets"),
        dataset_mode='test',
        dataset_list='test_pairs.txt',
        checkpoint_dir=CHECKPOINT_DIR,
        save_dir=RESULTS_DIR,
        grid_size=5
    )

    # Initialize models
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7  # Temporary change for ALIAS
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13  # Reset

    # Load weights
    load_checkpoint(seg, seg_ckpt_path)
    load_checkpoint(gmm, gmm_ckpt_path)
    load_checkpoint(alias, alias_ckpt_path)

    seg.eval()
    gmm.eval()
    alias.eval()

    return seg, gmm, alias, opt

def run_inference():
    """Core inference logic without any Streamlit components"""
    seg, gmm, alias, opt = load_models()
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))

    # Verify dataset paths
    if not os.path.exists(os.path.join(opt.dataset_dir, "test", "image")):
        raise FileNotFoundError(f"Missing model images in {opt.dataset_dir}/test/image")

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    if len(test_dataset) == 0:
        raise ValueError("No test pairs found in test_pairs.txt")

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            # ─── Processing Pipeline ──────────────────────────
            img_agnostic = inputs['img_agnostic']
            parse_agnostic = inputs['parse_agnostic']
            pose = inputs['pose']
            c = inputs['cloth']['unpaired']
            cm = inputs['cloth_mask']['unpaired']

            # ─── Segmentation ─────────────────────────────────
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size())), dim=1)
            
            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]

            # ─── Garment Warping ───────────────────────────────
            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float)
            parse_old.scatter_(1, parse_pred, 1.0)

            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float)
            parse[:, 0] = parse_old[:, 0]  # Background
            parse[:, 1] = torch.sum(parse_old[:, [2,4,7,8,9,10,11]], dim=1)  # Paste regions
            parse[:, 2] = parse_old[:, 3]  # Upper garment
            parse[:, 3] = parse_old[:, 1]  # Hair
            parse[:, 4] = parse_old[:, 5]  # Left arm
            parse[:, 5] = parse_old[:, 6]  # Right arm
            parse[:, 6] = parse_old[:, 12] # Noise

            # ─── GMM Alignment ─────────────────────────────────
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            # ─── Final Rendering ───────────────────────────────
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

            # ─── Save Results ──────────────────────────────────
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']
            unpaired_names = [f"{name.split('_')[0]}_{c_names[i]}" for i, name in enumerate(img_names)]
            
            save_images(output, unpaired_names, opt.save_dir)
            print(f"Processed batch {i+1}/{len(test_loader)}")

    print(f"Inference complete. Results saved to {opt.save_dir}")

if __name__ == "__main__":
    # Run directly when called by automated.py
    try:
        run_inference()
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        exit(1)
