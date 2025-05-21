import os
import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm

import gdown  # For downloading from Google Drive

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images


def download_if_not_exists(file_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {dest_path} from Google Drive...")
        gdown.download(url, dest_path, quiet=False)


# Replace argparse with hardcoded options
from types import SimpleNamespace

def get_opt():
    opt = SimpleNamespace()

    opt.name = 'run1'
    opt.batch_size = 1
    opt.workers = 0
    opt.load_height = 1024
    opt.load_width = 768
    opt.shuffle = False
    opt.dataset_dir = './datasets/'
    opt.dataset_mode = 'test'
    opt.dataset_list = 'test_pairs.txt'
    opt.checkpoint_dir = './checkpoints/'
    opt.save_dir = './results/'
    opt.display_freq = 1
    opt.seg_checkpoint = 'seg_final.pth'
    opt.gmm_checkpoint = 'gmm_final.pth'
    opt.alias_checkpoint = 'alias_final.pth'
    opt.semantic_nc = 13
    opt.init_type = 'xavier'
    opt.init_variance = 0.02
    opt.grid_size = 5
    opt.norm_G = 'spectralaliasinstance'
    opt.ngf = 64
    opt.num_upsampling_layers = 'most'

    return opt


def test(opt, seg, gmm, alias):
    opt.workers = 0

    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']

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

            save_images(output, unpaired_names, opt.save_dir)

            if (i + 1) % opt.display_freq == 0:
                print(f"step: {i + 1}")


def main():
    opt = get_opt()
    print("Options:", opt)

    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(opt.checkpoint_dir, exist_ok=True)

    # Google Drive file IDs
    seg_ckpt_id = "1Hb_y7M4pQlrKh6m4-2Mo_m_KU1IcT6DB"
    gmm_ckpt_id = "1gtagvr1I8Dq4ejnpQ51fZ9G9sCloKgyh"
    alias_ckpt_id = "1vWoDdaiWF0Zuv8Md9bn7q-xLUUIjXReY"

    # Local paths
    seg_ckpt_path = os.path.join(opt.checkpoint_dir, opt.seg_checkpoint)
    gmm_ckpt_path = os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint)
    alias_ckpt_path = os.path.join(opt.checkpoint_dir, opt.alias_checkpoint)

    # Download checkpoints if missing
    download_if_not_exists(seg_ckpt_id, seg_ckpt_path)
    download_if_not_exists(gmm_ckpt_id, gmm_ckpt_path)
    download_if_not_exists(alias_ckpt_id, alias_ckpt_path)

    # Load models
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

    test(opt, seg, gmm, alias)


if __name__ == '__main__':
    main()
