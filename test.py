# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch
import cv2
from PIL import Image
import yaml
from box import Box
import colorsys
import hashlib
import io
from tqdm import tqdm
# Imports from detany3d code
from train_utils import *
from wrap_model import WrapModel
import torch.nn as nn
import torch.distributed as dist
import random
from groundingdino.util.inference import load_model
from groundingdino.util.inference import predict as dino_predict
from torchvision.ops import box_convert
import groundingdino.datasets.transforms as T
# Add YOLO imports
from ultralytics import YOLO
# Disable distributed initialization
torch.distributed.is_available = lambda: False
torch.distributed.is_initialized = lambda: False
torch.distributed.get_world_size = lambda group=None: 1
torch.distributed.get_rank = lambda group=None: 0
logger = logging.getLogger("detany3d")
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)
# Define CATEGORY_NAMES and BOX_CLASS_INDEX (adapted from ovm3d, forcing 'box' for simplicity)
CATEGORY_NAMES = ['box'] # Force to 'box' similar to ovm3d
BOX_CLASS_INDEX = CATEGORY_NAMES.index('box')
def generate_image_token(image: np.ndarray) -> str:
    """Generate unique token based on image (SHA-256 hash)"""
    image = Image.fromarray(image)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    return hashlib.sha256(img_bytes.getvalue()).hexdigest()
def convert_image(img):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(img, 'RGB')
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed
def crop_hw(img):
    if img.dim() == 4:
        img = img.squeeze(0)
    h, w = img.shape[1:3]
    assert max(h, w) % 112 == 0, "target_size must be divisible by 112"
    new_h = (h // 14) * 14
    new_w = (w // 14) * 14
    center_h, center_w = h // 2, w // 2
    start_h = center_h - new_h // 2
    start_w = center_w - new_w // 2
    img_cropped = img[:, start_h:start_h + new_h, start_w:start_w + new_w]
    return img_cropped.unsqueeze(0)
def preprocess(x, cfg):
    sam_pixel_mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1)
    sam_pixel_std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1)
    x = (x - sam_pixel_mean) / sam_pixel_std
    h, w = x.shape[-2:]
    padh = cfg.model.pad - h
    padw = cfg.model.pad - w
    x = F.pad(x, (0, padw, 0, padh))
    return x
def preprocess_dino(x):
    x = x / 255
    IMAGENET_DATASET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    IMAGENET_DATASET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    x = (x - IMAGENET_DATASET_MEAN) / IMAGENET_DATASET_STD
    return x
def adjust_brightness(color, factor=1.5, v_min=0.3):
    r, g, b = color
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = max(v, v_min) * factor
    v = min(v, 1.0)
    return colorsys.hsv_to_rgb(h, s, v)
def draw_text(im, text, pos, scale=0.4, color='auto', font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255),
              blend=0.33, lineType=1):
    text = str(text)
    pos = [int(pos[0]), int(pos[1])]
    if color == 'auto':
        if bg_color is not None:
            color = (0, 0, 0) if ((bg_color[0] + bg_color[1] + bg_color[2])/3) > 127.5 else (255, 255, 255)
        else:
            color = (0, 0, 0)
    if bg_color is not None:
        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(x_s + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))
        im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0]*blend + bg_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1]*blend + bg_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2]*blend + bg_color[2] * (1 - blend)
        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))
    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)
def do_test(args, cfg, my_sam_model, dino_model, yolo_model):
    # Determine if input is a folder or a single file
    if os.path.isdir(args.input_folder):
        list_of_ims = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    elif os.path.isfile(args.input_folder) and args.input_folder.lower().endswith(('.jpg', '.png', '.jpeg')):
        list_of_ims = [args.input_folder]
    else:
        print(f"Invalid input: {args.input_folder} is not a directory or a valid image file.")
        return
    my_sam_model.eval()
    dino_model.eval()
    BOX_TRESHOLD = 0.7
    TEXT_TRESHOLD = 0.5
    output_dir = args.output_dir if args.output_dir else './exps/deploy'
    os.makedirs(output_dir, exist_ok=True)
    sam_trans = ResizeLongestSide(cfg.model.pad)
    for path in tqdm(list_of_ims):
        im_name = os.path.splitext(os.path.basename(path))[0]
        # Load image as BGR (for cv2), then convert to RGB
        im_bgr = cv2.imread(path)
        if im_bgr is None:
            continue
        img = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB) # RGB np.array
        # Run YOLO for 2D detections
        yolo_results = yolo_model.predict(source=path, conf=args.threshold, verbose=False)
        detections = yolo_results[0].boxes
        xyxy = detections.xyxy.cpu().numpy()
        confidences = detections.conf.cpu().numpy()
        class_ids = detections.cls.cpu().numpy().astype(int)
      
        num_dets = len(xyxy)
        if num_dets == 0:
            print(f'File: {im_name} with 0 dets from YOLO')
            cv2.imwrite(os.path.join(output_dir, im_name + '_boxes.jpg'), im_bgr)
            continue
        # Force all to 'box' similar to ovm3d
        bbox_2d_list = xyxy.astype(int).tolist()
        label_list = ['box'] * num_dets
        point_coords_list = [] # Empty to force mode='box'
        text = '' # No text prompt, use provided boxes
        # Processing adapted from predict function
        with torch.no_grad():
            if len(bbox_2d_list) > 0 and len(point_coords_list) > 0:
                print("Cannot handle bounding box and point at the same time. Skipping.")
                continue
            if len(point_coords_list) == 0:
                mode = 'box'
            else:
                mode = 'point'
                label_list = ["Unknown"]
            # No text and mode='box', so skip DINO and use YOLO boxes
            if text != '' and mode == 'point':
                print("Both text and point prompt input, following the point prompt")
            if text != '':
                image_source_dino, image_dino = convert_image(img)
                boxes, logits, phrases = dino_predict(
                    model=dino_model,
                    image=image_dino,
                    caption=text,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD,
                    remove_combined=False,
                )
                h, w = image_source_dino.shape[:2]
                boxes = boxes * torch.Tensor([w, h, w, h])
                xyxy_dino = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                for i, box in enumerate(xyxy_dino):
                    if mode == 'box':
                        bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
                        label_list.append(phrases[i])
                    elif mode == 'point':
                        pass
            if len(bbox_2d_list) == 0 and len(point_coords_list) == 0:
                print("No objects found in the image. Skipping.")
                continue
            raw_img = img.copy() # RGB np.array
            original_size = raw_img.shape[:2] # (h, w)
            img_torch = torch.from_numpy(raw_img).permute(2, 0, 1).float().unsqueeze(0)
            img_torch = sam_trans.apply_image_torch(img_torch)
            img_torch = crop_hw(img_torch)
            before_pad_size = tuple(img_torch.shape[2:])
            img_for_sam = preprocess(img_torch, cfg).to('cuda:0')
            img_for_dino = preprocess_dino(img_torch).to('cuda:0')
            if cfg.model.vit_pad_mask:
                vit_pad_size = (before_pad_size[0] // cfg.model.image_encoder.patch_size, before_pad_size[1] // cfg.model.image_encoder.patch_size)
            else:
                vit_pad_size = (cfg.model.pad // cfg.model.image_encoder.patch_size, cfg.model.pad // cfg.model.image_encoder.patch_size)
            if mode == 'box':
                bbox_2d_tensor = torch.tensor(bbox_2d_list)
                bbox_2d_tensor = sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).to('cuda:0')
                input_dict = {
                    "images": img_for_sam,
                    'vit_pad_size': torch.tensor(vit_pad_size).to('cuda:0').unsqueeze(0),
                    "images_shape": torch.Tensor(before_pad_size).to('cuda:0').unsqueeze(0),
                    "image_for_dino": img_for_dino,
                    "boxes_coords": bbox_2d_tensor,
                }
            elif mode == 'point':
                points_2d_tensor = torch.stack(point_coords_list, dim=1).to('cuda:0')
                points_2d_tensor = sam_trans.apply_coords_torch(points_2d_tensor, original_size)
                input_dict = {
                    "images": img_for_sam,
                    'vit_pad_size': torch.tensor(vit_pad_size).to('cuda:0').unsqueeze(0),
                    "images_shape": torch.Tensor(before_pad_size).to('cuda:0').unsqueeze(0),
                    "image_for_dino": img_for_dino,
                    "point_coords": points_2d_tensor,
                }
            ret_dict = my_sam_model(input_dict)
            K = ret_dict['pred_K']  # Predicted by model (fallback)

            print(K)
    
            # Override with custom intrinsics if provided
            if args.intrinsic:
                try:
                    fx, fy, cx, cy = map(float, args.intrinsic.split(','))
                    custom_K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().to('cuda:0').unsqueeze(0)
                    K = custom_K
                    print(f"Using custom camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                except ValueError:
                    print("Invalid intrinsic format; using predicted K. Expected: fx,fy,cx,cy")
    
            decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K)
            rot_mat = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])
            pred_box_ious = ret_dict.get('pred_box_ious', None)
            # Sample colors from image
            pixels = raw_img.reshape(-1, 3) / 255.0
            brightness = pixels.mean(axis=1)
            prob = brightness / brightness.sum()
            sampled_indices = np.random.choice(pixels.shape[0], min(100, pixels.shape[0]), p=prob, replace=False)
            sampled_colors = pixels[sampled_indices]
            sampled_colors = sorted(sampled_colors, key=lambda c: colorsys.rgb_to_hsv(*c)[2])
            adjusted_colors = [adjust_brightness(c, factor=2.0, v_min=0.4) for c in sampled_colors]
            img_slice = img_for_sam[0, :, :before_pad_size[0], :before_pad_size[1]].cpu()
            origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * img_slice + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
            todo = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
            K = K.detach().cpu().numpy()
            for i in range(len(decoded_bboxes_pred_2d)):
                if i >= len(adjusted_colors):
                    break # In case fewer colors than boxes
                x, y, z, w, h, l, yaw = decoded_bboxes_pred_3d[i].detach().cpu().numpy()
                rot_mat_i = rot_mat[i].detach().cpu().numpy()
                vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
                vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
                fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
                color = adjusted_colors[i]
                color = [min(255, c * 255) for c in color]
                best_j = torch.argmax(pred_box_ious[i])
                iou_score = pred_box_ious[i][best_j].item()
                draw_bbox_2d(todo, vertices_2d, color=(int(color[0]), int(color[1]), int(color[2])), thickness=3)
                if label_list[i] is not None:
                    draw_text(todo, f"{label_list[i]} {[round(c, 2) for c in decoded_bboxes_pred_3d[i][3:6].detach().cpu().numpy().tolist()]}", box_cxcywh_to_xyxy(decoded_bboxes_pred_2d[i]).detach().cpu().numpy().tolist(), scale=0.50 * todo.shape[0] / 500, bg_color=color)
            print(f'File: {im_name} with {len(decoded_bboxes_pred_2d)} dets')
            cv2.imwrite(os.path.join(output_dir, im_name + '_boxes.jpg'), todo)
def main(args):
    with open('./detect_anything/configs/demo.yaml', 'r', encoding='utf-8') as f:
        cfg_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = Box(cfg_dict)
    my_sam_model = WrapModel(cfg)
    checkpoint = torch.load(cfg.resume, map_location='cuda:0')
    new_model_dict = my_sam_model.state_dict()
    for k, v in new_model_dict.items():
        if k in checkpoint['state_dict'].keys() and checkpoint['state_dict'][k].size() == new_model_dict[k].size():
            new_model_dict[k] = checkpoint['state_dict'][k].detach()
    my_sam_model.load_state_dict(new_model_dict)
    my_sam_model.to('cuda:0')
    my_sam_model.setup()
    dino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
    dino_model.to('cuda:0')
    # Load YOLO model (adapt path as needed)
    yolo_model = YOLO('/root/ovmono3d/best.pt')
    with torch.no_grad():
        do_test(args, cfg, my_sam_model, dino_model, yolo_model)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input-folder', type=str, help='folder containing images to process', required=True)
    parser.add_argument('--labels-file', type=str, help='path to labels file (optional, not used)', default=None)
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for YOLO detections")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images (not implemented)")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory for results (default: ./exps/deploy)")
    parser.add_argument("--intrinsic", type=str, default=None, help="Custom camera intrinsics as fx,fy,cx,cy (overrides predicted K)")
    # Optional: Remove these if not needed
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend.",
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args) # Direct call instead of launch