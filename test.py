import os
import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
import gdown
import streamlit as st
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


def download_if_not_exists(file_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info(f"Downloading {os.path.basename(dest_path)} from Google Drive...")
        gdown.download(url, dest_path, quiet=False)


@st.cache_resource(show_spinner=False)
def load_models():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    seg_ckpt_path = os.path.join(CHECKPOINT_DIR, "seg_final.pth")
    gmm_ckpt_path = os.path.join(CHECKPOINT_DIR, "gmm_final.pth")
    alias_ckpt_path = os.path.join(CHECKPOINT_DIR, "alias_final.pth")

    download_if_not_exists(SEG_CKPT_ID, seg_ckpt_path)
    download_if_not_exists(GMM_CKPT_ID, gmm_ckpt_path)
    download_if_not_exists(ALIAS_CKPT_ID, alias_ckpt_path)

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
        dataset_dir='./datasets/',
        dataset_mode='test',
        dataset_list='test_pairs.txt',
        checkpoint_dir=CHECKPOINT_DIR,
        save_dir=RESULTS_DIR,
        display_freq=1,
        seg_checkpoint='seg_final.pth',
        gmm_checkpoint='gmm_final.pth',
        alias_checkpoint='alias_final.pth',
        grid_size=5
    )

    # Make sure dataset_list is an absolute path
    opt.dataset_list = os.path.join(opt.dataset_dir, opt.dataset_list)

    # Check test_pairs.txt existence and content
    if not os.path.exists(opt.dataset_list):
        st.error(f"ERROR: Dataset list file not found at {opt.dataset_list}")
    else:
        with open(opt.dataset_list, 'r') as f:
            lines = f.readlines()
        st.info(f"Found {len(lines)} test pairs in {opt.dataset_list}")
        st.write("Sample test pairs (first 5 lines):")
        for line in lines[:5]:
            st.write(line.strip())

    # Initialize models
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(seg, seg_ckpt_path)
    load_checkpoint(gmm, gmm_ckpt_path)
    load_checkpoint(alias, alias_ckpt_path)

    seg.eval()
    gmm.eval()
    alias.eval()

    return seg, gmm, alias, opt


def run_inference(opt, seg, gmm, alias):
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    total_batches = len(test_loader.data_loader)
    st.info(f"Total batches to process: {total_batches}")

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']

            st.write(f"Processing batch {i + 1}/{total_batches} - Image: {img_names[0]}, Cloth: {c_names[0]}")

            img_agnostic = inputs['img_agnostic']
            parse_agnostic = inputs['parse_agnostic']
            pose = inputs['pose']
            c = inputs['cloth']['unpaired']
            cm = inputs['cloth_mask']['unpaired']

            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size())), dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]

            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float)
            parse_old.scatter_(1, parse_pred, 1.0)

            labels = {
                0: ['background', [0]],
                1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
                2: ['upper', [3]],
                3: ['hair', [1]],
                4: ['left_arm', [5]],
                5: ['right_arm', [6]],
                6: ['noise', [12]]
            }
            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float)
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]

            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

            unpaired_names = []
            for img_name, c_name in zip(img_names, c_names):
                unpaired_names.append(f'{img_name.split("_")[0]}_{c_name}')

            save_images(output, unpaired_names, RESULTS_DIR)

    st.success(f"Inference done! Results saved to {RESULTS_DIR}")


def main():
    st.title("Virtual Try-On Demo")
    seg, gmm, alias, opt = load_models()

    st.write("Upload your test dataset folder with the correct structure (see docs).")

    if st.button("Run Virtual Try-On Inference"):
        with st.spinner("Running inference, please wait..."):
            try:
                run_inference(opt, seg, gmm, alias)
            except Exception as e:
                st.error(f"Error during inference: {e}")


if __name__ == "__main__":
    main()
