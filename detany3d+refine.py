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
import json  # Added for JSON output
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
# New visualization function
def visualize_boxes(bboxes_3d, rot_mat, decoded_bboxes_pred_2d, label_list, pred_box_ious, adjusted_colors, origin_img, K_np):
    todo = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    for i in range(len(decoded_bboxes_pred_2d)):
        if i >= len(adjusted_colors):
            break
        x, y, z, w, h, l, yaw = bboxes_3d[i].detach().cpu().numpy()
        rot_mat_i = rot_mat[i].detach().cpu().numpy()
        vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
        vertices_2d = project_to_image(vertices_3d, K_np.squeeze(0))
        fore_plane_center_2d = project_to_image(fore_plane_center_3d, K_np.squeeze(0))
        color = adjusted_colors[i]
        color = [min(255, c * 255) for c in color]
        best_j = torch.argmax(pred_box_ious[i])
        iou_score = pred_box_ious[i][best_j].item()
        draw_bbox_2d(todo, vertices_2d, color=(int(color[0]), int(color[1]), int(color[2])), thickness=3)
        if label_list[i] is not None:
            draw_text(todo, f"{label_list[i]} {[round(c, 2) for c in bboxes_3d[i][3:6].detach().cpu().numpy().tolist()]}", box_cxcywh_to_xyxy(decoded_bboxes_pred_2d[i]).detach().cpu().numpy().tolist(), scale=0.50 * todo.shape[0] / 500, bg_color=color)
    return todo
# New post-processing functions
def get_3d_box_corners(center, dims, rotation):
    dx, dy, dz = dims[0]/2, dims[1]/2, dims[2]/2
    corners_local = torch.tensor([
        [-dx, -dy, -dz], [dx, -dy, -dz], [-dx, dy, -dz], [dx, dy, -dz],
        [-dx, -dy, dz], [dx, -dy, dz], [-dx, dy, dz], [dx, dy, dz]
    ], dtype=torch.float32, device=center.device)
    corners = torch.mm(corners_local, rotation.T) + center.unsqueeze(0)
    return corners
def project_points_to_2d(points_3d, K, extrinsics=None):
    if extrinsics is None:
        extrinsics = torch.eye(4, device=points_3d.device)[:3]
    points_hom = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=1)
    points_cam = torch.mm(points_hom, extrinsics.T)
    projected = torch.mm(points_cam, K.T)
    projected_2d = projected[:, :2] / projected[:, 2:3].clamp(min=1e-6)
    return projected_2d
def project_3d_to_2d(corners_3d, K, extrinsics=None):
    projected_2d = project_points_to_2d(corners_3d, K, extrinsics)
    min_vals, _ = projected_2d.min(dim=0)
    max_vals, _ = projected_2d.max(dim=0)
    return torch.cat([min_vals, max_vals])
def compute_iou(box1, box2):
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou
def compute_containment_loss(proj_2d, gt_2d_box):
    overflow_left = torch.max(gt_2d_box[0] - proj_2d[0], torch.tensor(0.0, device=proj_2d.device))
    overflow_top = torch.max(gt_2d_box[1] - proj_2d[1], torch.tensor(0.0, device=proj_2d.device))
    overflow_right = torch.max(proj_2d[2] - gt_2d_box[2], torch.tensor(0.0, device=proj_2d.device))
    overflow_bottom = torch.max(proj_2d[3] - gt_2d_box[3], torch.tensor(0.0, device=proj_2d.device))
    return (overflow_left + overflow_top + overflow_right + overflow_bottom) / 4.0 # Average overflow
def compute_outside_cost(projected_2d, gt_2d_box):
    cost = 0.0
    for p in projected_2d:
        ox = max(0.0, gt_2d_box[0] - p[0]) + max(0.0, p[0] - gt_2d_box[2])
        oy = max(0.0, gt_2d_box[1] - p[1]) + max(0.0, p[1] - gt_2d_box[3])
        cost += ox + oy
    return cost
def refine_depth(pred_center, pred_dims, pred_rotation, K, gt_2d_box, extrinsics=None, iterations=500, lr=0.01):
    old_z = pred_center[2]
    if old_z <= 0:
        old_z = torch.tensor(1.0, device=pred_center.device) # avoid division by zero
    z = old_z.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    ray_dir = pred_center[:2] / old_z
    # Diagnostic initial computation
    current_center = torch.cat([ray_dir * z, z.unsqueeze(0)])
    corners_3d = get_3d_box_corners(current_center, pred_dims, pred_rotation)
    proj_2d = project_3d_to_2d(corners_3d, K, extrinsics)
    iou = compute_iou(proj_2d, gt_2d_box)
    containment_loss = compute_containment_loss(proj_2d, gt_2d_box)
    loss = 1 - iou + 0.5 * containment_loss # Weighted containment
    print(f"Initial proj_2d: {proj_2d}, gt_2d_box: {gt_2d_box}, iou: {iou.item()}, containment_loss: {containment_loss.item()}, dims: {pred_dims}")
    if loss.grad_fn is None:
        print("Skipping refinement: No gradient dependency detected for this box.")
        # Fallback with containment: Scale z to fit if overflowing
        if pred_dims[1] > 0.1:
            f = K[0, 0]
            h_pixel = gt_2d_box[3] - gt_2d_box[1]
            h_real = pred_dims[1]
            z_prior = (f * h_real) / h_pixel if h_pixel > 0 else old_z
            # Check if prior causes overflow; scale up if underfit, down if overfit
            new_center = torch.cat([ray_dir * z_prior, z_prior.unsqueeze(0)])
            temp_proj = project_3d_to_2d(get_3d_box_corners(new_center, pred_dims, pred_rotation), K, extrinsics)
            # if compute_containment_loss(temp_proj, gt_2d_box) > 0:
            # z_prior *= 1.2 # Increase z to shrink projection if overflowing
            # else:
            # z_prior *= 0.8 # Decrease z to expand projection if underfitting
            print(f"Applying size prior fallback with containment: new z = {z_prior}")
            return new_center
        return pred_center
    for _ in range(iterations):
        optimizer.zero_grad()
        current_center = torch.cat([ray_dir * z, z.unsqueeze(0)])
        corners_3d = get_3d_box_corners(current_center, pred_dims, pred_rotation)
        proj_2d = project_3d_to_2d(corners_3d, K, extrinsics)
        iou = compute_iou(proj_2d, gt_2d_box)
        containment_loss = compute_containment_loss(proj_2d, gt_2d_box)
        loss = 1 - iou + 0.5 * containment_loss
        loss.backward()
        optimizer.step()
    refined_z = z.detach()
    refined_center = torch.cat([ray_dir * refined_z, refined_z.unsqueeze(0)])
    return refined_center
def matrix_to_quaternion(m):
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return {"x": float(x), "y": float(y), "z": float(z), "w": float(w)}
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
            json_path_detany = os.path.join(output_dir, im_name + '_detany3d.json')
            with open(json_path_detany, 'w') as f:
                json.dump({"detections": []}, f, indent=4)
            json_path_refined = os.path.join(output_dir, im_name + '_refined.json')
            with open(json_path_refined, 'w') as f:
                json.dump({"detections": []}, f, indent=4)
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
            K = ret_dict['pred_K'] # Predicted by model (fallback)
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
            # Sample colors from image (shared)
            pixels = raw_img.reshape(-1, 3) / 255.0
            brightness = pixels.mean(axis=1)
            prob = brightness / brightness.sum()
            sampled_indices = np.random.choice(pixels.shape[0], min(100, pixels.shape[0]), p=prob, replace=False)
            sampled_colors = pixels[sampled_indices]
            sampled_colors = sorted(sampled_colors, key=lambda c: colorsys.rgb_to_hsv(*c)[2])
            adjusted_colors = [adjust_brightness(c, factor=2.0, v_min=0.4) for c in sampled_colors]
            # Prepare origin_img (shared)
            img_slice = img_for_sam[0, :, :before_pad_size[0], :before_pad_size[1]].cpu()
            origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * img_slice + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
            K_np = K.detach().cpu().numpy()
            # Save JSON for detany3d only (before refinement)
            detections_detany = []
            for i in range(len(decoded_bboxes_pred_2d)):
                if i >= len(adjusted_colors):
                    break
                x, y, z, w, h, l, yaw = decoded_bboxes_pred_3d[i].detach().cpu().numpy()
                rot_mat_i = rot_mat[i].detach().cpu().numpy()
                color = adjusted_colors[i]
                det = {
                    "category": label_list[i],
                    "score": 1.0,
                    "center_cam": [float(x), float(y), float(z)],
                    "dimensions": [float(w), float(h), float(l)],
                    "pose1": rot_mat_i[0].tolist(),
                    "pose2": rot_mat_i[1].tolist(),
                    "pose3": rot_mat_i[2].tolist(),
                    "color": [float(color[0]), float(color[1]), float(color[2])]
                }
                detections_detany.append(det)
            json_path_detany = os.path.join(output_dir, im_name + '_detany3d.json')
            with open(json_path_detany, 'w') as f:
                json.dump({"detections": detections_detany}, f, indent=4)
            # Visualize BEFORE refinement
            before_img = visualize_boxes(decoded_bboxes_pred_3d.clone(), rot_mat, decoded_bboxes_pred_2d, label_list, pred_box_ious, adjusted_colors, origin_img, K_np)
            # Do depth refinement
            K_torch = K.squeeze(0) # (3,3) tensor on cuda
            for i in range(len(decoded_bboxes_pred_3d)):
                original_center = decoded_bboxes_pred_3d[i][:3].clone()
                dims = decoded_bboxes_pred_3d[i][3:6]
                rotation = rot_mat[i]
                gt_2d_box = torch.tensor(bbox_2d_list[i], dtype=torch.float32, device='cuda:0') # xyxy
                refined_center = refine_depth(original_center, dims, rotation, K_torch, gt_2d_box)
                # Compute outside costs for both
                corners_3d_original = get_3d_box_corners(original_center, dims, rotation)
                projected_2d_original = project_points_to_2d(corners_3d_original, K_torch)
                original_cost = compute_outside_cost(projected_2d_original, gt_2d_box)
                corners_3d_refined = get_3d_box_corners(refined_center, dims, rotation)
                projected_2d_refined = project_points_to_2d(corners_3d_refined, K_torch)
                refined_cost = compute_outside_cost(projected_2d_refined, gt_2d_box)
                # Choose the one with lower cost
                if refined_cost < original_cost:
                    decoded_bboxes_pred_3d[i][:3] = refined_center
                # else keep original
            # Save JSON for refine+detany3d (after refinement)
            detections_refined = []
            for i in range(len(decoded_bboxes_pred_3d)):
                if i >= len(adjusted_colors):
                    break
                x, y, z, w, h, l, yaw = decoded_bboxes_pred_3d[i].detach().cpu().numpy()
                rot_mat_i = rot_mat[i].detach().cpu().numpy()
                color = adjusted_colors[i]
                det = {
                    "category": label_list[i],
                    "score": 1.0,
                    "center_cam": [float(x), float(y), float(z)],
                    "dimensions": [float(w), float(h), float(l)],
                    "pose1": rot_mat_i[0].tolist(),
                    "pose2": rot_mat_i[1].tolist(),
                    "pose3": rot_mat_i[2].tolist(),
                    "color": [float(color[0]), float(color[1]), float(color[2])]
                }
                detections_refined.append(det)
            json_path_refined = os.path.join(output_dir, im_name + '_refined.json')
            with open(json_path_refined, 'w') as f:
                json.dump({"detections": detections_refined}, f, indent=4)
            # Visualize AFTER all refinements
            after_img = visualize_boxes(decoded_bboxes_pred_3d, rot_mat, decoded_bboxes_pred_2d, label_list, pred_box_ious, adjusted_colors, origin_img, K_np)
            # Combine into single image (side-by-side)
            combined_img = np.hstack((before_img, after_img))
            cv2.putText(combined_img, "Before", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_img, "After (Depth)", (before_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(output_dir, im_name + '_comparison.jpg'), combined_img)
            print(f'File: {im_name} with {len(decoded_bboxes_pred_2d)} dets')
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
    parser.add_argument("--threshold", type=float, default=0.7, help="threshold on score for YOLO detections")
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