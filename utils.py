import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise


def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        try:
            array = tensor.numpy().astype('uint8')
        except:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        im = Image.fromarray(array)
        im.save(os.path.join(save_dir, img_name), format='JPEG')


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path))


# --------------------------
# OpenPose Keypoint Generation
# --------------------------
def generate_openpose_keypoints(image_path, output_path):
    """
    Generates OpenPose keypoints for the given image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save generated keypoints.

    Returns:
        str: Path to the generated keypoints JSON file.
    """
    if os.path.exists(output_path):
        print(f"OpenPose keypoints already exist: {output_path}")
        return output_path  # Return existing file if present

    # Run OpenPose (make sure OpenPose is installed)
    openpose_cmd = f'openpose --image_dir {os.path.dirname(image_path)} --write_json {output_path} --display 0 --render_pose 0'
    os.system(openpose_cmd)

    if os.path.exists(output_path):
        print(f"Generated OpenPose keypoints: {output_path}")
        return output_path
    else:
        raise RuntimeError("Failed to generate OpenPose keypoints.")


# --------------------------
# Cloth Mask Generation
# --------------------------
def generate_cloth_mask(cloth_image_path, mask_output_path):
    """
    Generates a binary cloth mask for a given clothing image.

    Args:
        cloth_image_path (str): Path to the input clothing image.
        mask_output_path (str): Path to save the generated mask.

    Returns:
        str: Path to the saved cloth mask.
    """
    if os.path.exists(mask_output_path):
        print(f"Cloth mask already exists: {mask_output_path}")
        return mask_output_path  # Return existing mask

    # Read image
    cloth = cv2.imread(cloth_image_path, cv2.IMREAD_UNCHANGED)

    if cloth is None:
        raise ValueError(f"Cannot load image: {cloth_image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary mask
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Save mask
    cv2.imwrite(mask_output_path, mask)

    print(f"Generated cloth mask: {mask_output_path}")
    return mask_output_path
