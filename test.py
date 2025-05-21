import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/')

    parser.add_argument('--display_freq', type=int, default=1)

    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth')
    parser.add_argument('--alias_checkpoint', type=str, default='alias_final.pth')

    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type',
                        choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'],
                        default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02)

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5)

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--num_upsampling_layers',
                        choices=['normal', 'more', 'most'],
                        default='most')

    opt = parser.parse_args()
    return opt

def test(opt, seg, gmm, alias):
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    if len(test_dataset) == 0:
        raise RuntimeError("No test pairs found. Did you update test_pairs.txt correctly?")

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['img_name']
            c_names   = inputs['c_name']['unpaired']

            img_agnostic  = inputs['img_agnostic']
            parse_agnostic = inputs['parse_agnostic']
            pose           = inputs['pose']
            c              = inputs['cloth']['unpaired']
            cm             = inputs['cloth_mask']['unpaired']

            # — Part 1: Segmentation generation —
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down           = F.interpolate(pose,           size=(256, 192), mode='bilinear')
            c_masked_down       = F.interpolate(c * cm,         size=(256, 192), mode='bilinear')
            cm_down             = F.interpolate(cm,            size=(256, 192), mode='bilinear')

            seg_input = torch.cat((
                cm_down,
                c_masked_down,
                parse_agnostic_down,
                pose_down,
                gen_noise(cm_down.size())
            ), dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred      = gauss(up(parse_pred_down)).argmax(dim=1)[:, None]

            # Build the 13→7 channel semantic map
            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float)
            parse_old.scatter_(1, parse_pred, 1.0)

            # Collapse into 7 categories
            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float)
            parse[:, 0] = parse_old[:, 0]                                           # background
            parse[:, 1] = torch.sum(parse_old[:, [2,4,7,8,9,10,11]], dim=1)         # "paste" regions
            parse[:, 2] = parse_old[:, 3]                                           # upper
            parse[:, 3] = parse_old[:, 1]                                           # hair
            parse[:, 4] = parse_old[:, 5]                                           # left_arm
            parse[:, 5] = parse_old[:, 6]                                           # right_arm
            parse[:, 6] = parse_old[:, 12]                                          # noise

            # — Part 2: Clothes Deformation (GMM) —
            agnostic_gmm    = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3],      size=(256, 192), mode='nearest')
            pose_gmm        = F.interpolate(pose,               size=(256, 192), mode='nearest')
            c_gmm           = F.interpolate(c,                  size=(256, 192), mode='nearest')

            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
            _, warped_grid = gmm(gmm_input, c_gmm)

            warped_c  = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            # — Part 3: Try-on synthesis (ALIAS) —
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias(
                torch.cat((img_agnostic, pose, warped_c), dim=1),
                parse,
                parse_div,
                misalign_mask
            )

            # Save the results into ./results/
            unpaired_names = [f"{img_names[k].split('_')[0]}_{c_names[k]}" for k in range(len(img_names))]
            save_images(output, unpaired_names, opt.save_dir)

            if (i + 1) % opt.display_freq == 0:
                print(f"Processed batch {i+1}/{len(test_loader)}")

def main():
    opt = get_opt()
    print(f"Options: {opt}")

    # Ensure output folder exists
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)

    # Initialize models (CPU only)
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt,       inputA_nc=7,          inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    # Load weights from ./checkpoints/
    load_checkpoint(seg,   os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm,   os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

    seg.eval()
    gmm.eval()
    alias.eval()

    try:
        test(opt, seg, gmm, alias)
        print("✅ test.py finished without exceptions.")
    except Exception as e:
        print(f"❌ test.py encountered an error:\n    {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
