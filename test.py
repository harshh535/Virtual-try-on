import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import gdown
from types import SimpleNamespace

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images

# ─── Google Drive checkpoint IDs ───────────────────────────────────────────────
SEG_CKPT_ID   = "1Hb_y7M4pQlrKh6m4-2Mo_m_KU1IcT6DB"
GMM_CKPT_ID   = "1gtagvr1I8Dq4ejnpQ51fZ9G9sCloKgyh"
ALIAS_CKPT_ID = "1vWoDdaiWF0Zuv8Md9bn7q-xLUUIjXReY"

CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR    = "./results"

def download_if_not_exists(file_id, dest_path):
    """
    If `dest_path` does not exist, download from Google Drive using gdown.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"⬇️ Downloading {os.path.basename(dest_path)}...")
        gdown.download(url, dest_path, quiet=False)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)

    parser.add_argument('--dataset_dir',   type=str, default='./datasets')
    parser.add_argument('--dataset_list',  type=str, default='test/test_pairs.txt')
    parser.add_argument('--save_dir',      type=str, default='./results')
    parser.add_argument('--checkpoint_dir',type=str, default='./checkpoints')
    parser.add_argument('--load_height',   type=int, default=1024)
    parser.add_argument('--load_width',    type=int, default=768)
    parser.add_argument('--semantic_nc',   type=int, default=13)
    parser.add_argument('--grid_size',     type=int, default=5)
    parser.add_argument('--norm_G',        type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf',           type=int, default=64)
    parser.add_argument('--num_upsampling_layers', choices=['normal','more','most'], default='most')
    parser.add_argument('--display_freq',  type=int, default=1)
    return parser.parse_args()

def load_models(opt):
    """
    Download checkpoints if necessary, construct networks, load weights, and return them.
    """
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)

    seg_path   = os.path.join(opt.checkpoint_dir, "seg_final.pth")
    gmm_path   = os.path.join(opt.checkpoint_dir, "gmm_final.pth")
    alias_path = os.path.join(opt.checkpoint_dir, "alias_final.pth")

    download_if_not_exists(SEG_CKPT_ID,   seg_path)
    download_if_not_exists(GMM_CKPT_ID,   gmm_path)
    download_if_not_exists(ALIAS_CKPT_ID, alias_path)

    # Instantiate networks
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)

    prev_semantic = opt.semantic_nc
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = prev_semantic

    # Load pretrained weights
    load_checkpoint(seg,   seg_path)
    load_checkpoint(gmm,   gmm_path)
    load_checkpoint(alias, alias_path)

    seg.eval()
    gmm.eval()
    alias.eval()

    return seg, gmm, alias

def run_inference(opt, seg, gmm, alias):
    """
    Core inference logic: iterate over test pairs, warp the cloth, and render final try-ons.
    Saves output images under opt.save_dir.
    """
    up    = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))

    image_folder = os.path.join(opt.dataset_dir, "test", "image")
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Missing model-images folder: {image_folder}")

    test_dataset = VITONDataset(opt)
    test_loader  = VITONDataLoader(opt, test_dataset)

    if len(test_dataset) == 0:
        raise ValueError("⚠️ No test pairs found in test/test_pairs.txt")

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_agnostic   = inputs['img_agnostic']
            parse_agnostic = inputs['parse_agnostic']
            pose           = inputs['pose']
            c              = inputs['cloth']['unpaired']
            cm             = inputs['cloth_mask']['unpaired']

            # ── Part 1: Segmentation prediction ────────────────────────────────
            seg_input = torch.cat([
                F.interpolate(cm,           (256, 192), mode='bilinear'),
                F.interpolate(c * cm,       (256, 192), mode='bilinear'),
                F.interpolate(parse_agnostic,(256, 192), mode='bilinear'),
                F.interpolate(pose,         (256, 192), mode='bilinear'),
                gen_noise((c.size(0),1,256,192))
            ], dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred      = gauss(up(parse_pred_down)).argmax(dim=1)[:, None]

            # Convert to a 13-channel one-hot map
            parse_old = torch.zeros(parse_pred.size(0), opt.semantic_nc, opt.load_height, opt.load_width).to(parse_pred.device)
            parse_old.scatter_(1, parse_pred, 1.0)

            # Combine into 7 semantic channels for GMM input
            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width).to(parse_pred.device)
            parse[:, 0] = parse_old[:, 0]                              # background
            parse[:, 1] = parse_old[:, [2,4,7,8,9,10,11]].sum(dim=1)   # paste regions
            parse[:, 2] = parse_old[:, 3]                              # upper garment
            parse[:, 3] = parse_old[:, 1]                              # hair
            parse[:, 4] = parse_old[:, 5]                              # left arm
            parse[:, 5] = parse_old[:, 6]                              # right arm
            parse[:, 6] = parse_old[:, 12]                             # noise

            # ── Part 2: Warp cloth with GMM ──────────────────────────────────
            agnostic_gmm     = F.interpolate(img_agnostic, (256, 192), mode='nearest')
            parse_cloth_gmm  = F.interpolate(parse[:, 2:3], (256, 192), mode='nearest')
            pose_gmm         = F.interpolate(pose, (256, 192), mode='nearest')
            c_gmm            = F.interpolate(c, (256, 192), mode='nearest')
            gmm_input        = torch.cat([parse_cloth_gmm, pose_gmm, agnostic_gmm], dim=1)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c       = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm      = F.grid_sample(cm, warped_grid, padding_mode='border')

            # ── Part 3: Final rendering via ALIAS ─────────────────────────────
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask = torch.clamp(misalign_mask, min=0.0)
            parse_div     = torch.cat([parse, misalign_mask], dim=1)
            parse_div[:, 2:3] -= misalign_mask

            rendered = alias(
                torch.cat([img_agnostic, pose, warped_c], dim=1),
                parse,
                parse_div,
                misalign_mask
            )

            img_names = inputs['img_name']
            c_names   = inputs['c_name']['unpaired']
            save_names = [f"{name.split('_')[0]}_{c_names[j]}" for j, name in enumerate(img_names)]
            save_images(rendered, save_names, opt.save_dir)

            print(f"✅ Processed batch {i+1}/{len(test_loader)}")

    print(f"\n🎉 Inference complete! Results saved to {opt.save_dir}")

def main():
    opt = get_opt()
    os.makedirs(opt.save_dir, exist_ok=True)
    seg, gmm, alias = load_models(opt)
    run_inference(opt, seg, gmm, alias)

if __name__ == "__main__":
    main()
