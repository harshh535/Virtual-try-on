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

# Google Drive file IDs
SEG_CKPT_ID = "1Hb_y7M4pQlrKh6m4-2Mo_m_KU1IcT6DB"
GMM_CKPT_ID = "1gtagvr1I8Dq4ejnpQ51fZ9G9sCloKgyh"
ALIAS_CKPT_ID = "1vWoDdaiWF0Zuv8Md9bn7q-xLUUIjXReY"

CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

def download_checkpoint(file_id, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"📦 Downloading {os.path.basename(dest_path)}...")
        gdown.download(url, dest_path, quiet=False)

def load_models():
    """Load and return pretrained models and config"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Checkpoint paths
    seg_ckpt = os.path.join(CHECKPOINT_DIR, "seg_final.pth")
    gmm_ckpt = os.path.join(CHECKPOINT_DIR, "gmm_final.pth")
    alias_ckpt = os.path.join(CHECKPOINT_DIR, "alias_final.pth")

    # Download if missing
    download_checkpoint(SEG_CKPT_ID, seg_ckpt)
    download_checkpoint(GMM_CKPT_ID, gmm_ckpt)
    download_checkpoint(ALIAS_CKPT_ID, alias_ckpt)

    # Config
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
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    # Load checkpoints
    load_checkpoint(seg, seg_ckpt)
    load_checkpoint(gmm, gmm_ckpt)
    load_checkpoint(alias, alias_ckpt)

    seg.eval()
    gmm.eval()
    alias.eval()

    return seg, gmm, alias, opt

def run_inference():
    seg, gmm, alias, opt = load_models()

    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))

    # Dataset validation
    image_dir = os.path.join(opt.dataset_dir, "test", "image")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    if len(test_dataset) == 0:
        raise ValueError("⚠️ No test pairs found in test_pairs.txt!")

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_agnostic = inputs['img_agnostic']
            parse_agnostic = inputs['parse_agnostic']
            pose = inputs['pose']
            c = inputs['cloth']['unpaired']
            cm = inputs['cloth_mask']['unpaired']

            # Step 1: Segmentation Prediction
            seg_input = torch.cat([
                F.interpolate(cm, (256, 192), mode='bilinear'),
                F.interpolate(c * cm, (256, 192), mode='bilinear'),
                F.interpolate(parse_agnostic, (256, 192), mode='bilinear'),
                F.interpolate(pose, (256, 192), mode='bilinear'),
                gen_noise((c.size(0), 1, 256, 192))
            ], dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down)).argmax(dim=1)[:, None]

            # Convert to semantic map
            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width).to(parse_pred.device)
            parse_old.scatter_(1, parse_pred, 1.0)

            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width).to(parse_pred.device)
            parse[:, 0] = parse_old[:, 0]
            parse[:, 1] = parse_old[:, [2, 4, 7, 8, 9, 10, 11]].sum(dim=1)
            parse[:, 2] = parse_old[:, 3]
            parse[:, 3] = parse_old[:, 1]
            parse[:, 4] = parse_old[:, 5]
            parse[:, 5] = parse_old[:, 6]
            parse[:, 6] = parse_old[:, 12]

            # Step 2: Warping with GMM
            gmm_input = torch.cat([
                F.interpolate(parse[:, 2:3], (256, 192), mode='nearest'),
                F.interpolate(pose, (256, 192), mode='nearest'),
                F.interpolate(img_agnostic, (256, 192), mode='nearest')
            ], dim=1)

            _, warped_grid = gmm(gmm_input, F.interpolate(c, (256, 192), mode='nearest'))
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            # Step 3: Final Rendering
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask = torch.clamp(misalign_mask, min=0.0)
            parse_div = torch.cat([parse, misalign_mask], dim=1)
            parse_div[:, 2:3] -= misalign_mask

            rendered = alias(torch.cat([img_agnostic, pose, warped_c], dim=1), parse, parse_div, misalign_mask)

            # Save
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']
            save_names = [f"{name.split('_')[0]}_{c_names[i]}" for i, name in enumerate(img_names)]
            save_images(rendered, save_names, opt.save_dir)

            print(f"✅ Processed batch {i+1}/{len(test_loader)}")

    print(f"\n🎉 Inference complete! Results saved to {opt.save_dir}")

if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
