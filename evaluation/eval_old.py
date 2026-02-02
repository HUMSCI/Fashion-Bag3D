#!/usr/bin/env python3
import argparse
import glob
import numpy as np
import os
from typing import Optional, Tuple, List, Dict

from PIL import Image
from scipy import linalg
import torch
from sklearn.linear_model import LinearRegression
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from tqdm import tqdm

# NEW: for PSNR/SSIM
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# your originals
import utils
import inception
import image_metrics
import json
import random
import pandas as pd

ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
CKPT_URL = 'https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth'

# -------------------------
# Dataset helpers (yours)
# -------------------------

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, device='cpu', num_workers=1):
    """Computes the activations of for all images.

    Args:
        files (list): List of image file paths.
        model (torch.nn.Module): Model for computing activations.
        batch_size (int): Batch size for computing activations.
        device (torch.device): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (): Activations of the images, shape [num_images, 2048].
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=Compose([Resize((512,512)),ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), 2048))

    start_idx = 0

    pbar = tqdm(total=len(files))
    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            features = model(batch, return_features=True)

        features = features.cpu().numpy()
        pred_arr[start_idx:start_idx + features.shape[0]] = features
        start_idx = start_idx + features.shape[0]

        pbar.update(batch.shape[0])

    pbar.close()
    return pred_arr


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1); sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f'fid calc produces singular product; adding {eps} to diagonal')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_activation_statistics(files, model, batch_size=50, device='cpu', num_workers=1):
    """Computes the activation statistics used by the FID.
    
    Args:
        files (list): List of image file paths.
        model (torch.nn.Module): Model for computing activations.
        batch_size (int): Batch size for computing activations.
        device (torch.device): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (np.ndarray, np.ndarray): mean of activations, covariance of activations
        
    """
    act = get_activations(files, model, batch_size, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_image_paths(path, sort=False):
    """Returns the paths of the images in the specified directory, filtered by allowed file extensions.

    Args:
        path (str): Path to image directory.
        sort (bool): Sort paths alphanumerically.

    Returns:
        (list): List of image paths with allowed file extensions.

    """
    paths = []
    for extension in ALLOWED_IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(path, f'*.{extension}')))
    if sort:
        paths.sort()
    return paths


#modm


def compute_fid(stylized_image_paths, style_image_paths, batch_size, device, num_workers=1):
    """Computes the FID for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    ckpt_file = utils.download(CKPT_URL)
    ckpt = torch.load(ckpt_file, map_location=device)
    
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    # stylized_image_paths = get_image_paths(path_to_stylized)
    # style_image_paths = get_image_paths(path_to_style)

    mu1, sigma1 = compute_activation_statistics(stylized_image_paths, model, batch_size, device, num_workers)
    mu2, sigma2 = compute_activation_statistics(style_image_paths, model, batch_size, device, num_workers)
    
    fid_value = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value


def compute_fid_infinity(stylized_image_paths, style_image_paths, batch_size, device, num_points=15, num_workers=1):
    """Computes the FID infinity for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        num_points (int): Number of FID_N we evaluate to fit a line.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID infinity value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    ckpt_file = utils.download(CKPT_URL)
    ckpt = torch.load(ckpt_file, map_location=device)
    
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # stylized_image_paths = get_image_paths(path_to_stylized)
    # style_image_paths = get_image_paths(path_to_style)

    assert len(stylized_image_paths) == len(style_image_paths), \
           f'Number of stylized images and number of style images must be equal.({len(stylized_image_paths)},{len(style_image_paths)})'

    activations_stylized = get_activations(stylized_image_paths, model, batch_size, device, num_workers)
    activations_style = get_activations(style_image_paths, model, batch_size, device, num_workers)
    activation_idcs = np.arange(activations_stylized.shape[0])

    fids = []
    
    fid_batches = np.linspace(start=5000, stop=len(stylized_image_paths), num=num_points).astype('int32')
    
    for fid_batch_size in fid_batches:
        np.random.shuffle(activation_idcs)
        idcs = activation_idcs[:fid_batch_size]
        
        act_style_batch = activations_style[idcs]
        act_stylized_batch = activations_stylized[idcs]

        mu_style, sigma_style = np.mean(act_style_batch, axis=0), np.cov(act_style_batch, rowvar=False)
        mu_stylized, sigma_stylized = np.mean(act_stylized_batch, axis=0), np.cov(act_stylized_batch, rowvar=False)
        
        fid_value = compute_frechet_distance(mu_style, sigma_style, mu_stylized, sigma_stylized)
        fids.append(fid_value)

    fids = np.array(fids).reshape(-1, 1)
    reg = LinearRegression().fit(1 / fid_batches.reshape(-1, 1), fids)
    fid_infinity = reg.predict(np.array([[0]]))[0,0]

    return fid_infinity

def compute_art_fid(path_to_stylized, path_to_style, path_to_content, batch_size, device, mode='art_fid_inf', content_metric='lpips', num_workers=1):
    """Computes the FID for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        path_to_content (str): Path to the content images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet'
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) ArtFID value.
    """
    print('Compute FID value...')
    if mode == 'art_fid_inf':
        fid_value = compute_fid_infinity(path_to_stylized, path_to_style, batch_size, device, num_workers)
    elif mode == 'art_fid':
        fid_value = compute_fid(path_to_stylized, path_to_style, batch_size, device, num_workers)
    elif mode == 'style_loss':
        fid_value = compute_style_loss(path_to_stylized, path_to_style, batch_size, device, num_workers)
    else:
        fid_value = compute_gram_loss(path_to_stylized, path_to_style, batch_size, device, num_workers)
    
    print('Compute content distance...')
    cnt_value = compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric, device, num_workers)
    gray_cnt_value = compute_content_distance(path_to_stylized, path_to_content, batch_size, content_metric, device, num_workers, gray=True)

    art_fid_value = (cnt_value + 1) * (fid_value + 1)
    # fid_value = f'{fid_value.item():.4f}'
    # cnt_value = f'{content_dist.item():.4f}'
    # gray_cnt_value = f'{gray_content_dist.item():.4f}'
    # art_fid_value = (cnt_value + 1) * (fid_value + 1)
    return art_fid_value.item(), fid_value.item(), cnt_value.item(), gray_cnt_value.item(), 

def compute_cfsd(path_to_stylized, path_to_content, batch_size, device, num_workers=1):
    """Computes CFSD for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_content (str): Path to the content images.
        batch_size (int): Batch size for computing activations.
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) CFSD value.
    """
    print('Compute CFSD value...')

    simi_val = compute_patch_simi(path_to_stylized, path_to_content, 1, device, num_workers)
    simi_dist = f'{simi_val.item():.4f}'
    return simi_dist



def compute_content_distance(stylized_image_paths, content_image_paths, batch_size, content_metric='lpips', device='cuda', num_workers=1, gray=False):
    """Computes the distance for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet'
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # Sort paths in order to match up the stylized images with the corresponding content image
    # stylized_image_paths = get_image_paths(path_to_stylized, sort=True)
    # content_image_paths = get_image_paths(path_to_content, sort=True)

    assert len(stylized_image_paths) == len(content_image_paths), \
           'Number of stylized images and number of content images must be equal.'

    if gray:
        content_transforms = Compose([Resize((512,512)), Grayscale(),
        ToTensor()])
    else:
        content_transforms = Compose([Resize((512,512)),
        ToTensor()])
    
    dataset_stylized = ImagePathDataset(stylized_image_paths, transforms=content_transforms)
    dataloader_stylized = torch.utils.data.DataLoader(dataset_stylized,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=num_workers)

    dataset_content = ImagePathDataset(content_image_paths, transforms=content_transforms)
    dataloader_content = torch.utils.data.DataLoader(dataset_content,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)
    
    metric_list = ['alexnet', 'ssim', 'ms-ssim']
    if content_metric in metric_list:
        metric = image_metrics.Metric(content_metric).to(device)
    elif content_metric == 'lpips':
        metric = image_metrics.LPIPS().to(device)
    elif content_metric == 'vgg':
        metric = image_metrics.LPIPS_vgg().to(device)
    else:
        raise ValueError(f'Invalid content metric: {content_metric}')

    dist_sum = 0.0
    N = 0
    pbar = tqdm(total=len(stylized_image_paths))
    for batch_stylized, batch_content in zip(dataloader_stylized, dataloader_content):
        with torch.no_grad():
            batch_dist = metric(batch_stylized.to(device), batch_content.to(device))
            N += batch_stylized.shape[0]
            dist_sum += torch.sum(batch_dist)

        pbar.update(batch_stylized.shape[0])

    pbar.close()

    return dist_sum / N

def compute_patch_simi(stylized_image_paths, content_image_paths, batch_size, device, num_workers=1):
    """Computes the distance for the given paths.

    Args:
        path_to_stylized (str): Path to the stylized images.
        path_to_style (str): Path to the style images.
        batch_size (int): Batch size for computing activations.
        content_metric (str): Metric to use for content distance. Choices: 'lpips', 'vgg', 'alexnet'
        device (str): Device for commputing activations.
        num_workers (int): Number of threads for data loading.

    Returns:
        (float) FID value.
    """
    device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    # Sort paths in order to match up the stylized images with the corresponding content image
    # stylized_image_paths = get_image_paths(path_to_stylized, sort=True)
    # content_image_paths = get_image_paths(path_to_content, sort=True)

    assert len(stylized_image_paths) == len(content_image_paths), \
           'Number of stylized images and number of content images must be equal.'

    style_transforms = ToTensor()
    
    dataset_stylized = ImagePathDataset(stylized_image_paths, transforms=style_transforms)
    dataloader_stylized = torch.utils.data.DataLoader(dataset_stylized,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=num_workers)

    dataset_content = ImagePathDataset(content_image_paths, transforms=style_transforms)
    dataloader_content = torch.utils.data.DataLoader(dataset_content,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=num_workers)
    
    metric = image_metrics.PatchSimi(device=device).to(device)

    dist_sum = 0.0
    N = 0
    pbar = tqdm(total=len(stylized_image_paths))
    for batch_stylized, batch_content in zip(dataloader_stylized, dataloader_content):
        with torch.no_grad():
            batch_dist = metric(batch_stylized.to(device), batch_content.to(device))
            N += batch_stylized.shape[0]
            dist_sum += torch.sum(batch_dist)

        pbar.update(batch_stylized.shape[0])

    pbar.close()

    return dist_sum / N

# ---------------------------------
# FID/KID wrappers using your model
# ---------------------------------
def _load_inception(device):
    device = torch.device('cuda') if (device == 'cuda' and torch.cuda.is_available()) else torch.device('cpu')
    ckpt_file = utils.download(CKPT_URL)
    ckpt = torch.load(ckpt_file, map_location=device)
    model = inception.Inception3().to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model, device


def compute_fid_from_lists(gen_files: List[str], ref_files: List[str], batch_size: int, device: str, num_workers: int = 1) -> float:
    """FID using your Inception on two arbitrary file lists."""
    model, dev = _load_inception(device)
    mu1, sigma1 = compute_activation_statistics(gen_files, model, batch_size, dev, num_workers)
    mu2, sigma2 = compute_activation_statistics(ref_files, model, batch_size, dev, num_workers)
    return float(compute_frechet_distance(mu1, sigma1, mu2, sigma2))


def compute_kid_from_lists(gen_files: List[str], ref_files: List[str], batch_size: int, device: str, num_workers: int = 1,
                           subset_size: int = 100, num_subsets: int = 50) -> Tuple[float, float]:
    """
    KID (Kernel Inception Distance) computed from Inception activations with unbiased MMD.
    Returns (mean, std) across random subsets for stability.
    """
    model, dev = _load_inception(device)
    feats_g = get_activations(gen_files, model, batch_size, dev, num_workers)
    feats_r = get_activations(ref_files, model, batch_size, dev, num_workers)

    if len(feats_g) == 0 or len(feats_r) == 0:
        return float('nan'), float('nan')

    d = feats_g.shape[1]
    def poly_kernel(x, y):  # degree-3 polynomial kernel k(x,y) = ((x·y)/d + 1)^3
        return (x @ y.T / d + 1.0) ** 3

    def mmd_unbiased(X, Y):
        # X: [n, d], Y: [m, d]
        n = X.shape[0]; m = Y.shape[0]
        Kxx = poly_kernel(X, X)
        Kyy = poly_kernel(Y, Y)
        Kxy = poly_kernel(X, Y)
        # remove diagonals for unbiased estimate
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        term_x = Kxx.sum() / (n * (n - 1))
        term_y = Kyy.sum() / (m * (m - 1))
        term_xy = 2.0 * Kxy.mean()
        return float(term_x + term_y - term_xy)

    rng = np.random.default_rng(1234)
    n = min(len(feats_g), len(feats_r))
    ss = min(subset_size, n)
    vals = []
    for _ in range(num_subsets):
        idx_g = rng.choice(len(feats_g), ss, replace=False)
        idx_r = rng.choice(len(feats_r), ss, replace=False)
        vals.append(mmd_unbiased(feats_g[idx_g], feats_r[idx_r]))
    vals = np.array(vals, dtype=np.float64)

    SCALE = 100.0
    return float(vals.mean()) * SCALE, float(vals.std()) * SCALE

    # return float(vals.mean()), float(vals.std())


# ---------------------------------
# Pairing logic for front/back views
# ---------------------------------
def _find_first_in(dirpath: str, name_hint: str) -> Optional[str]:
    """front_view/back_view filename finder (exact or contains)."""
    # exact
    for ext in ALLOWED_IMAGE_EXTENSIONS:
        p = os.path.join(dirpath, f"{name_hint}.{ext}")
        if os.path.isfile(p):
            return p
    # contains
    hint = name_hint.lower()
    cands = []
    for f in os.listdir(dirpath):
        if any(f.endswith(f".{e}") for e in ALLOWED_IMAGE_EXTENSIONS):
            stem = os.path.splitext(f)[0].lower()
            if hint in stem:
                cands.append(os.path.join(dirpath, f))
    cands.sort()
    return cands[0] if cands else None


def collect_pairs(ref_root: str, gen_root: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns dict: {"front": [(ref_path, gen_path), ...], "back": [...]}
    Only includes pairs where both sides exist.
    Expects per-bag subfolders under ref_root and gen_root with matching names.
    """
    pairs = {"front": [], "back": []}
    bag_dirs = sorted([p for p in glob.glob(os.path.join(ref_root, "*")) if os.path.isdir(p)])
    for ref_dir in bag_dirs:
        bag = os.path.basename(ref_dir.rstrip("/"))
        ref_front = _find_first_in(ref_dir, "front_view")
        ref_back  = _find_first_in(ref_dir, "back_view")

        gen_dir = os.path.join(gen_root, bag)
        if not os.path.isdir(gen_dir):
            continue
        gen_front = _find_first_in(gen_dir, "front_view")
        gen_back  = _find_first_in(gen_dir, "back_view")

        if ref_front and gen_front:
            pairs["front"].append((ref_front, gen_front))
        if ref_back and gen_back:
            pairs["back"].append((ref_back, gen_back))
    return pairs


def collect_pairs_from_json(ref_root: str, gen_root: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns dict: {"front": [(ref_path, gen_path), ...], "back": [...]}
    Only includes pairs where both sides exist.
    Expects per-bag subfolders under ref_root and gen_root with matching names.

    For the reference side (ref_root), the function now reads labels.json inside
    each bag folder to determine which images are "front_view" and "back_view".
    """
    pairs = {"front": [], "back": []}

    # all bag folders under ref_root
    bag_dirs = sorted(
        [p for p in glob.glob(os.path.join(ref_root, "*")) if os.path.isdir(p)]
    )

    for ref_dir in bag_dirs:
        bag = os.path.basename(ref_dir.rstrip("/"))

        # --- read labels.json in the ref folder ---
        labels_path = os.path.join(ref_dir, "labels.json")
        if not os.path.isfile(labels_path):
            # no labels.json -> skip this bag
            # print(f"[skip] {bag}: no labels.json")
            continue

        try:
            with open(labels_path, "r") as f:
                labels = json.load(f)
        except json.JSONDecodeError:
            # broken labels.json -> skip
            print(f"[warn] {bag}: cannot parse {labels_path}, skipping")
            continue

        front_name = labels.get("front_view")
        back_name  = labels.get("back_view")

        # Build full paths for reference images from labels.json
        ref_front = os.path.join(ref_dir, front_name) if front_name else None
        ref_back  = os.path.join(ref_dir, back_name)  if back_name  else None

        # Make sure files actually exist
        if ref_front and not os.path.isfile(ref_front):
            print(f"[warn] {bag}: front_view file not found: {ref_front}")
            ref_front = None
        if ref_back and not os.path.isfile(ref_back):
            print(f"[warn] {bag}: back_view file not found: {ref_back}")
            ref_back = None

        # --- generated side (still search by substring) ---
        gen_dir = os.path.join(gen_root, bag)
        if not os.path.isdir(gen_dir):
            # no generated folder -> skip
            continue

        gen_front = _find_first_in(gen_dir, "front_view")
        gen_back  = _find_first_in(gen_dir, "back_view")

        # --- only add pairs if both ref + gen exist for that view ---
        if ref_front and gen_front:
            pairs["front"].append((ref_front, gen_front))
        if ref_back and gen_back:
            pairs["back"].append((ref_back, gen_back))

    return pairs


# ---------------------------------
# Pairwise metrics (PSNR/SSIM/LPIPS)
# ---------------------------------
def _resize_like(img_src: Image.Image, img_tgt: Image.Image) -> Image.Image:
    return img_src.resize(img_tgt.size, Image.BICUBIC)


def _psnr_ssim_pair(gen_img: Image.Image, gt_img: Image.Image) -> Tuple[float, float]:
    if gen_img.size != gt_img.size:
        gen_img = _resize_like(gen_img, gt_img)
    A = np.array(gt_img, dtype=np.float32)
    B = np.array(gen_img, dtype=np.float32)
    psnr = psnr_metric(A, B, data_range=255.0)
    ssim = ssim_metric(A, B, data_range=255.0, channel_axis=2)
    return float(psnr), float(ssim)


def compute_pairwise_metrics(pairs: List[Tuple[str, str]], device: str = 'cuda') -> Dict[str, float]:
    """
    Mean PSNR / SSIM / LPIPS over matched pairs.
    We reuse your image_metrics implementations for LPIPS/SSIM if desired,
    but here PSNR/SSIM are computed via skimage and LPIPS via image_metrics for consistency with your stack.
    """
    if len(pairs) == 0:
        return {"psnr": np.nan, "ssim": np.nan, "lpips": np.nan}

    device = torch.device('cuda') if (device == 'cuda' and torch.cuda.is_available()) else torch.device('cpu')

    ref_files = [r for (r, g) in pairs]
    gen_files = [g for (r, g) in pairs]
    
    artfid, fid, lpips, lpips_gray = compute_art_fid(gen_files,
                                            ref_files,
                                            ref_files,
                                            16,
                                            device,
                                            mode='art_fid',
                                            content_metric='lpips',
                                            )
    

        # LPIPS via your wrapper (expects tensors in [0,1], typically resized)
    lpips_metric = image_metrics.LPIPS().to(device).eval()

    psnrs, ssims, lpipss = [], [], []
    # transform used for LPIPS (to keep behavior similar to your compute_content_distance)
    lpips_transform = Compose([Resize((512,512)), ToTensor()])

    for ref_path, gen_path in tqdm(pairs, desc="Pairwise (PSNR/SSIM/LPIPS)"):
        gt_img  = Image.open(ref_path).convert('RGB')
        gen_img = Image.open(gen_path).convert('RGB')

        # PSNR/SSIM on native GT size (gen->gt resized)
        p, s = _psnr_ssim_pair(gen_img, gt_img)
        psnrs.append(p); ssims.append(s)

        # LPIPS: resize both with same transform to be consistent
        A = lpips_transform(gen_img).unsqueeze(0).to(device)
        B = lpips_transform(gt_img).unsqueeze(0).to(device)
        with torch.no_grad():
            d = lpips_metric(A, B).mean().item()
        lpipss.append(d)

    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        # "lpips": float(np.mean(lpipss)),
        "artfid": artfid, ##modm
        "fid": fid,
        "lpips": lpips,
        "lpips_gray": lpips_gray,
    }


# ---------------------------------
# Top-level runner using your API
# ---------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute PSNR, SSIM, LPIPS, FID, KID for bag renders vs GT (front/back).")
    parser.add_argument('--ref_root', type=str, default='/mnt/data/mdm/datasets/bag/farfetch/women/category_high_labeled/Tote_Bags',
                        help='GT (reference) root with per-bag subfolders.')
    parser.add_argument('--gen_root', type=str, default='/mnt/process/mdm/bag/mdm_output/farfetch/women/kiss3dgen/Tote_Bags/image_to_3d',
                        help='Rendered images root with per-bag subfolders.')
    # /mnt/process/mdm/bag/mdm_output/farfetch/women/kiss3dgen/Tote_Bags/image_to_3d
    # /mnt/process/mdm/bag/mdm_output/farfetch/women/trellis_glb/Tote_Bags_mv
    
    ##kiss3dgen: /mnt/process/mdm/bag/kiss3dgen/image_to_3d
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for Inception activations.')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers for activations.')
    parser.add_argument('--device', type=str, default='cuda:1', choices=['cuda', 'cpu'], help='Device for metrics that use torch.')
    # KID stability
    parser.add_argument('--kid_subset', type=int, default=100, help='Subset size per KID estimate.')
    parser.add_argument('--kid_subsets', type=int, default=50, help='Number of KID subsets to average.')

    args = parser.parse_args()

    # 1) collect matched pairs
    # pairs = collect_pairs(args.ref_root, args.gen_root) #front_view, back_view directly from the folders
    pairs = collect_pairs_from_json(args.ref_root, args.gen_root) #front_view, back_view from labels.json
    front_pairs = pairs["front"]
    back_pairs  = pairs["back"]

    print(f"Matched pairs: front={len(front_pairs)}  back={len(back_pairs)}")


    # 2) pairwise metrics
    front_pairwise = compute_pairwise_metrics(front_pairs, device=args.device)
    back_pairwise  = compute_pairwise_metrics(back_pairs,  device=args.device) if back_pairs else {"psnr": np.nan, "ssim": np.nan, "lpips": np.nan}

    # 3) set-level metrics (FID/KID) using your Inception
    front_ref_files = [r for (r, g) in front_pairs]
    front_gen_files = [g for (r, g) in front_pairs]
    back_ref_files  = [r for (r, g) in back_pairs]
    back_gen_files  = [g for (r, g) in back_pairs]

    front_fid = compute_fid_from_lists(front_gen_files, front_ref_files, args.batch_size, args.device, args.num_workers) if len(front_pairs) else float('nan')
    back_fid  = compute_fid_from_lists(back_gen_files,  back_ref_files,  args.batch_size, args.device, args.num_workers)  if len(back_pairs)  else float('nan')

    front_kid_mean, front_kid_std = compute_kid_from_lists(front_gen_files, front_ref_files, args.batch_size, args.device,
                                                           args.num_workers, subset_size=args.kid_subset, num_subsets=args.kid_subsets) if len(front_pairs) else (float('nan'), float('nan'))
    back_kid_mean, back_kid_std   = compute_kid_from_lists(back_gen_files,  back_ref_files,  args.batch_size, args.device,
                                                           args.num_workers, subset_size=args.kid_subset, num_subsets=args.kid_subsets) if len(back_pairs) else (float('nan'), float('nan'))

    # 4) print summary
    print("\n=== Front view ===")
    print(f"PSNR (↑): {front_pairwise['psnr']:.4f}")
    print(f"SSIM (↑): {front_pairwise['ssim']:.4f}")
    print(f"LPIPS (↓): {front_pairwise['lpips']:.4f}")
    print(f"FID (↓): {front_fid:.4f}")
    print(f"KID mean±std (↓): {front_kid_mean:.4f} ± {front_kid_std:.4f}")

    print("\n=== Back view ===")
    if len(back_pairs) > 0:
        print(f"PSNR (↑): {back_pairwise['psnr']:.4f}")
        print(f"SSIM (↑): {back_pairwise['ssim']:.4f}")
        print(f"LPIPS (↓): {back_pairwise['lpips']:.4f}")
        print(f"FID (↓): {back_fid:.4f}")
        print(f"KID mean±std (↓): {back_kid_mean:.4f} ± {back_kid_std:.4f}")
    else:
        print("No back-view GT+render pairs found; skipping.")


## save reulsts on the average
def main_mul_cat():
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, LPIPS, FID, KID for bag renders vs GT (front/back) over multiple categories."
    )
    parser.add_argument(
        '--ref_root_base',
        type=str,
        default='/mnt/data/mdm/datasets/bag/farfetch/women/category_high_labeled',
        help='GT (reference) base root with per-category subfolders.'
    )
    parser.add_argument(
        '--gen_root_base',
        type=str,
        default='/mnt/process/mdm/bag/mdm_output/farfetch/women/kiss3dgen',
        help='Rendered images base root with per-category subfolders.'
    )
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for Inception activations.')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers for activations.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for metrics that use torch.')
    parser.add_argument('--kid_subset', type=int, default=100, help='Subset size per KID estimate.')
    parser.add_argument('--kid_subsets', type=int, default=50, help='Number of KID subsets to average.')
    parser.add_argument('--out_xlsx', type=str, default='bag_metrics_kiss3dgen.xlsx', help='Output Excel file.')


    # Farfetch categories to evaluate
    FARFETCH_CATEGORIES = [
        # "Backpacks",
        # "Beach_Bags",
        # "Belt_Bags",
        # "Bucket_Bags",
        # "Clutch_Bags",
        # "Luggage",
        # "Messenger_&_Crossbody_Bags",
        # "Mini_Bags",
        # "Shoulder_Bags",
        "Tote_Bags",
    ]


    args = parser.parse_args()

    front_psnr_list, front_ssim_list, front_lpips_list = [], [], []
    front_fid_list, front_kid_list = [], []

    back_psnr_list, back_ssim_list, back_lpips_list = [], [], []
    back_fid_list, back_kid_list = [], []

    for cat in FARFETCH_CATEGORIES:
        ref_root = os.path.join(args.ref_root_base, cat)
        # for kiss3dgen: <gen_root_base>/<Category>/image_to_3d/<bag>
        gen_root = os.path.join(args.gen_root_base, cat, "image_to_3d")

        if not os.path.isdir(ref_root):
            print(f"[skip] Missing ref category folder: {ref_root}")
            continue
        if not os.path.isdir(gen_root):
            print(f"[skip] Missing gen category folder: {gen_root}")
            continue

        print(f"\n=== Category: {cat} ===")
        print(f"ref_root: {ref_root}")
        print(f"gen_root: {gen_root}")

        # 1) collect matched pairs from labels.json
        pairs = collect_pairs_from_json(ref_root, gen_root)
        front_pairs = pairs["front"]
        back_pairs  = pairs["back"]

        print(f"Matched pairs: front={len(front_pairs)}  back={len(back_pairs)}")
        if len(front_pairs) == 0 and len(back_pairs) == 0:
            print("[warn] No pairs found for this category, skipping.")
            continue

        # 2) pairwise metrics (PSNR/SSIM/LPIPS + artFID/FID inside)
        front_pairwise = compute_pairwise_metrics(front_pairs, device=args.device) if len(front_pairs) else {"psnr": np.nan, "ssim": np.nan, "lpips": np.nan}
        back_pairwise  = compute_pairwise_metrics(back_pairs,  device=args.device)  if len(back_pairs)  else {"psnr": np.nan, "ssim": np.nan, "lpips": np.nan}

        # 3) set-level metrics (FID/KID) using your Inception wrapper
        front_ref_files = [r for (r, g) in front_pairs]
        front_gen_files = [g for (r, g) in front_pairs]
        back_ref_files  = [r for (r, g) in back_pairs]
        back_gen_files  = [g for (r, g) in back_pairs]

        front_fid = compute_fid_from_lists(front_gen_files, front_ref_files, args.batch_size, args.device, args.num_workers) if len(front_pairs) else float('nan')
        back_fid  = compute_fid_from_lists(back_gen_files,  back_ref_files,  args.batch_size, args.device, args.num_workers)  if len(back_pairs)  else float('nan')

        front_kid_mean, _ = compute_kid_from_lists(
            front_gen_files, front_ref_files, args.batch_size, args.device,
            args.num_workers, subset_size=args.kid_subset, num_subsets=args.kid_subsets
        ) if len(front_pairs) else (float('nan'), float('nan'))

        back_kid_mean, _ = compute_kid_from_lists(
            back_gen_files, back_ref_files, args.batch_size, args.device,
            args.num_workers, subset_size=args.kid_subset, num_subsets=args.kid_subsets
        ) if len(back_pairs) else (float('nan'), float('nan'))

        # 4) accumulate per-category results
        if len(front_pairs):
            front_psnr_list.append(front_pairwise["psnr"])
            front_ssim_list.append(front_pairwise["ssim"])
            front_lpips_list.append(front_pairwise["lpips"])
            front_fid_list.append(front_fid)
            front_kid_list.append(front_kid_mean)

        if len(back_pairs):
            back_psnr_list.append(back_pairwise["psnr"])
            back_ssim_list.append(back_pairwise["ssim"])
            back_lpips_list.append(back_pairwise["lpips"])
            back_fid_list.append(back_fid)
            back_kid_list.append(back_kid_mean)

        # optional: print per-category summary
        print("\n  [Category summary: front]")
        if len(front_pairs):
            print(f"    PSNR: {front_pairwise['psnr']:.4f}, SSIM: {front_pairwise['ssim']:.4f}, LPIPS: {front_pairwise['lpips']:.4f}, FID: {front_fid:.4f}, KID: {front_kid_mean:.4f}")
        else:
            print("    No front-view pairs.")

        print("  [Category summary: back]")
        if len(back_pairs):
            print(f"    PSNR: {back_pairwise['psnr']:.4f}, SSIM: {back_pairwise['ssim']:.4f}, LPIPS: {back_pairwise['lpips']:.4f}, FID: {back_fid:.4f}, KID: {back_kid_mean:.4f}")
        else:
            print("    No back-view pairs.")

    # 5) average across all categories (ignoring NaN)
    def nanmean(lst):
        arr = np.array(lst, dtype=np.float64)
        return float(np.nanmean(arr)) if np.isfinite(arr).any() else float('nan')

    front_avg_psnr  = nanmean(front_psnr_list)
    front_avg_ssim  = nanmean(front_ssim_list)
    front_avg_lpips = nanmean(front_lpips_list)
    front_avg_fid   = nanmean(front_fid_list)
    front_avg_kid   = nanmean(front_kid_list)

    back_avg_psnr  = nanmean(back_psnr_list)
    back_avg_ssim  = nanmean(back_ssim_list)
    back_avg_lpips = nanmean(back_lpips_list)
    back_avg_fid   = nanmean(back_fid_list)
    back_avg_kid   = nanmean(back_kid_list)

    print("\n================ Overall averages across categories (kiss3dgen) ================")
    print("Front view:")
    print(f"  PSNR: {front_avg_psnr:.4f}, SSIM: {front_avg_ssim:.4f}, LPIPS: {front_avg_lpips:.4f}, FID: {front_avg_fid:.4f}, KID: {front_avg_kid:.4f}")
    print("Back view:")
    print(f"  PSNR: {back_avg_psnr:.4f}, SSIM: {back_avg_ssim:.4f}, LPIPS: {back_avg_lpips:.4f}, FID: {back_avg_fid:.4f}, KID: {back_avg_kid:.4f}")

    # 6) save to Excel in the requested format
    rows = [
        {
            "view": "front_view",
            "PSNR": front_avg_psnr,
            "SSIM": front_avg_ssim,
            "LPIPS": front_avg_lpips,
            "FID": front_avg_fid,
            "KID": front_avg_kid,
        },
        {
            "view": "back_view",
            "PSNR": back_avg_psnr,
            "SSIM": back_avg_ssim,
            "LPIPS": back_avg_lpips,
            "FID": back_avg_fid,
            "KID": back_avg_kid,
        },
    ]
    df = pd.DataFrame(rows, columns=["view", "PSNR", "SSIM", "LPIPS", "FID", "KID"])
    df.to_excel(args.out_xlsx, index=False)
    print(f"\n[info] Saved averaged metrics to: {args.out_xlsx}")



## save reulsts for each category, and the average
def main_mul_cat_detail():
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, LPIPS, FID, KID for bag renders vs GT (front/back) over multiple categories."
    )
    parser.add_argument(
        '--ref_root_base',
        type=str,
        default='/mnt/data/mdm/datasets/bag/farfetch/women/category_high_labeled',
        help='GT (reference) base root with per-category subfolders.'
    )
    parser.add_argument(
        '--gen_root_base',
        type=str,
        # default='/mnt/process/mdm/bag/mdm_output/farfetch/women/kiss3dgen', ##kiss3dgen
        default='/mnt/process/mdm/bag/mdm_output/farfetch/women/trellis_glb', ##trellis_glb
        # default='/mnt/process/mdm/bag/mdm_output/farfetch/women/trellis_glb', ##ours
        # default='/mnt/process/mdm/bag/mdm_output/farfetch/women/instantMesh', ##instantMesh
        # default='/mnt/process/mdm/bag/mdm_output/farfetch/women/3Dtopia-XL', ##3Dtopia-XL
        help='Rendered images base root with per-category subfolders.'
    )
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for Inception activations.')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers for activations.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for metrics that use torch.')
    parser.add_argument('--kid_subset', type=int, default=100, help='Subset size per KID estimate.')
    parser.add_argument('--kid_subsets', type=int, default=50, help='Number of KID subsets to average.')
    parser.add_argument('--out_xlsx', type=str, default='bag_metrics_trellis_detail.xlsx', help='Output Excel file.')

    # Farfetch categories to evaluate
    FARFETCH_CATEGORIES = [
        "Backpacks",
        "Beach_Bags",
        "Belt_Bags",
        "Bucket_Bags",
        "Clutch_Bags",
        "Luggage",
        "Messenger_&_Crossbody_Bags",
        "Mini_Bags",
        "Shoulder_Bags",
        "Tote_Bags",
    ]

    args = parser.parse_args()

    # for computing averages
    front_psnr_list, front_ssim_list, front_lpips_list = [], [], []
    front_fid_list, front_kid_list = [], []

    back_psnr_list, back_ssim_list, back_lpips_list = [], [], []
    back_fid_list, back_kid_list = [], []

    # for saving per-category results (ordered as requested)
    rows_front = []
    rows_back = []
    MAX_PAIRS = 100

    for cat in FARFETCH_CATEGORIES:
        ref_root = os.path.join(args.ref_root_base, cat) 

        ##generation
        # gen_root = os.path.join(args.gen_root_base, cat, "image_to_3d") ##kissgemn3d
        gen_root = os.path.join(args.gen_root_base, cat) ##trellis_glb
        # gen_root = os.path.join(args.gen_root_base, cat+'_mv') ##ours
        # gen_root = os.path.join(args.gen_root_base, cat, "instant-mesh-large","meshes") ##instantMesh
        # gen_root = os.path.join(args.gen_root_base, cat, "inference", "3dtopia-xl-sview") ##3Dtopia-XL

        if not os.path.isdir(ref_root):
            print(f"[skip] Missing ref category folder: {ref_root}")
            continue
        if not os.path.isdir(gen_root):
            print(f"[skip] Missing gen category folder: {gen_root}")
            continue

        print(f"\n=== Category: {cat} ===")
        print(f"ref_root: {ref_root}")
        print(f"gen_root: {gen_root}")

        # 1) collect matched pairs from labels.json
        pairs = collect_pairs_from_json(ref_root, gen_root)
        front_pairs = pairs["front"]
        back_pairs  = pairs["back"]

        print(f"Matched pairs (before cap): front={len(front_pairs)}  back={len(back_pairs)}")

        # NEW: limit to first MAX_PAIRS pairs per view
        if len(front_pairs) > MAX_PAIRS:
            front_pairs = front_pairs[:MAX_PAIRS]
        if len(back_pairs) > MAX_PAIRS:
            back_pairs = back_pairs[:MAX_PAIRS]

        print(f"Using pairs (after cap):   front={len(front_pairs)}  back={len(back_pairs)}")

        if len(front_pairs) == 0 and len(back_pairs) == 0:
            print("[warn] No pairs found for this category, skipping.")
            continue

        # 2) pairwise metrics (PSNR/SSIM/LPIPS + artFID/FID inside)
        front_pairwise = compute_pairwise_metrics(front_pairs, device=args.device) if len(front_pairs) else {"psnr": np.nan, "ssim": np.nan, "lpips": np.nan}
        back_pairwise  = compute_pairwise_metrics(back_pairs,  device=args.device)  if len(back_pairs)  else {"psnr": np.nan, "ssim": np.nan, "lpips": np.nan}

        # 3) set-level metrics (FID/KID) using your Inception wrapper
        front_ref_files = [r for (r, g) in front_pairs]
        front_gen_files = [g for (r, g) in front_pairs]
        back_ref_files  = [r for (r, g) in back_pairs]
        back_gen_files  = [g for (r, g) in back_pairs]

        front_fid = compute_fid_from_lists(front_gen_files, front_ref_files, args.batch_size, args.device, args.num_workers) if len(front_pairs) else float('nan')
        back_fid  = compute_fid_from_lists(back_gen_files,  back_ref_files,  args.batch_size, args.device, args.num_workers)  if len(back_pairs)  else float('nan')

        front_kid_mean, _ = compute_kid_from_lists(
            front_gen_files, front_ref_files, args.batch_size, args.device,
            args.num_workers, subset_size=args.kid_subset, num_subsets=args.kid_subsets
        ) if len(front_pairs) else (float('nan'), float('nan'))

        back_kid_mean, _ = compute_kid_from_lists(
            back_gen_files, back_ref_files, args.batch_size, args.device,
            args.num_workers, subset_size=args.kid_subset, num_subsets=args.kid_subsets
        ) if len(back_pairs) else (float('nan'), float('nan'))

        # 4) accumulate per-category results for averaging
        if len(front_pairs):
            front_psnr_list.append(front_pairwise["psnr"])
            front_ssim_list.append(front_pairwise["ssim"])
            front_lpips_list.append(front_pairwise["lpips"])
            front_fid_list.append(front_fid)
            front_kid_list.append(front_kid_mean)

        if len(back_pairs):
            back_psnr_list.append(back_pairwise["psnr"])
            back_ssim_list.append(back_pairwise["ssim"])
            back_lpips_list.append(back_pairwise["lpips"])
            back_fid_list.append(back_fid)
            back_kid_list.append(back_kid_mean)

        # 5) store per-category rows separately for front and back (ordering control)
        if len(front_pairs):
            rows_front.append({
                "category": cat,
                "view": "front_view",
                "PSNR": front_pairwise["psnr"],
                "SSIM": front_pairwise["ssim"],
                "LPIPS": front_pairwise["lpips"],
                "FID": front_fid,
                "KID": front_kid_mean,
            })

        if len(back_pairs):
            rows_back.append({
                "category": cat,
                "view": "back_view",
                "PSNR": back_pairwise["psnr"],
                "SSIM": back_pairwise["ssim"],
                "LPIPS": back_pairwise["lpips"],
                "FID": back_fid,
                "KID": back_kid_mean,
            })

        # optional: print per-category summary
        print("\n  [Category summary: front]")
        if len(front_pairs):
            print(f"    PSNR: {front_pairwise['psnr']:.4f}, SSIM: {front_pairwise['ssim']:.4f}, LPIPS: {front_pairwise['lpips']:.4f}, FID: {front_fid:.4f}, KID: {front_kid_mean:.4f}")
        else:
            print("    No front-view pairs.")

        print("  [Category summary: back]")
        if len(back_pairs):
            print(f"    PSNR: {back_pairwise['psnr']:.4f}, SSIM: {back_pairwise['ssim']:.4f}, LPIPS: {back_pairwise['lpips']:.4f}, FID: {back_fid:.4f}, KID: {back_kid_mean:.4f}")
        else:
            print("    No back-view pairs.")

    # 6) average across all categories (ignoring NaN)
    def nanmean(lst):
        arr = np.array(lst, dtype=np.float64)
        return float(np.nanmean(arr)) if np.isfinite(arr).any() else float('nan')

    front_avg_psnr  = nanmean(front_psnr_list)
    front_avg_ssim  = nanmean(front_ssim_list)
    front_avg_lpips = nanmean(front_lpips_list)
    front_avg_fid   = nanmean(front_fid_list)
    front_avg_kid   = nanmean(front_kid_list)

    back_avg_psnr  = nanmean(back_psnr_list)
    back_avg_ssim  = nanmean(back_ssim_list)
    back_avg_lpips = nanmean(back_lpips_list)
    back_avg_fid   = nanmean(back_fid_list)
    back_avg_kid   = nanmean(back_kid_list)

    print("\n================ Overall averages across categories (kiss3dgen, capped at 100 pairs/view/category) ================")
    print("Front view:")
    print(f"  PSNR: {front_avg_psnr:.4f}, SSIM: {front_avg_ssim:.4f}, LPIPS: {front_avg_lpips:.4f}, FID: {front_avg_fid:.4f}, KID: {front_avg_kid:.4f}")
    print("Back view:")
    print(f"  PSNR: {back_avg_psnr:.4f}, SSIM: {back_avg_ssim:.4f}, LPIPS: {back_avg_lpips:.4f}, FID: {back_avg_fid:.4f}, KID: {back_avg_kid:.4f}")

    # 7) build average rows
    rows_avg = [
        {
            "category": "ALL",
            "view": "front_view",
            "PSNR": front_avg_psnr,
            "SSIM": front_avg_ssim,
            "LPIPS": front_avg_lpips,
            "FID": front_avg_fid,
            "KID": front_avg_kid,
        },
        {
            "category": "ALL",
            "view": "back_view",
            "PSNR": back_avg_psnr,
            "SSIM": back_avg_ssim,
            "LPIPS": back_avg_lpips,
            "FID": back_avg_fid,
            "KID": back_avg_kid,
        },
    ]

    # 8) concatenate in desired order: all front, then all back, then averages
    excel_rows = rows_front + rows_back + rows_avg

    df = pd.DataFrame(excel_rows, columns=["category", "view", "PSNR", "SSIM", "LPIPS", "FID", "KID"])
    df.to_excel(args.out_xlsx, index=False)
    print(f"\n[info] Saved per-category and average metrics to: {args.out_xlsx}")


# def main_mul_cat_detail_10run():
#     parser = argparse.ArgumentParser(
#         description="Compute PSNR, SSIM, LPIPS, FID, KID for bag renders vs GT (front/back) over multiple categories with repeated random sampling."
#     )
#     parser.add_argument(
#         '--ref_root_base',
#         type=str,
#         default='/mnt/data/mdm/datasets/bag/farfetch/women/category_high_labeled',
#         help='GT (reference) base root with per-category subfolders.'
#     )
#     #['instantMesh','trellis_glb','kiss3dgen','3Dtopia-XL','Ours']
#     parser.add_argument('--method', type=str, default='instantMesh', help='A list of methods') 
#     parser.add_argument('--batch_size', type=int, default=4, help='Batch size for Inception activations.')
#     parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers for activations.')
#     parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for metrics that use torch.')
#     parser.add_argument('--kid_subset', type=int, default=100, help='Subset size per KID estimate.')
#     parser.add_argument('--kid_subsets', type=int, default=50, help='Number of KID subsets to average.')
#     parser.add_argument('--out_xlsx', type=str, default='results/trellis_glb.xlsx', help='Output Excel file.')

#     # Farfetch categories to evaluate
#     FARFETCH_CATEGORIES = [
#         "Backpacks",
#         "Beach_Bags",
#         "Belt_Bags",
#         "Bucket_Bags",
#         "Clutch_Bags",
#         "Luggage",
#         "Messenger_&_Crossbody_Bags",
#         "Mini_Bags",
#         "Shoulder_Bags",
#         "Tote_Bags",
#     ]

#     args = parser.parse_args()

#     ## define the method root
#     #['instantMesh','trellis_glb','kiss3dgen','3Dtopia-XL','Ours']
#     args.gen_root_base = '/mnt/process/mdm/bag/mdm_output/farfetch/women/'+args.method
#     args.out_xlsx = 'results/' + args.method +'.xlsx'

#     # for computing averages over categories
#     front_psnr_list, front_ssim_list, front_lpips_list = [], [], []
#     front_fid_list, front_kid_list = [], []

#     back_psnr_list, back_ssim_list, back_lpips_list = [], [], []
#     back_fid_list, back_kid_list = [], []

#     # for saving per-category results (ordered as requested)
#     rows_front = []
#     rows_back = []

#     MAX_PAIRS = 100
#     N_RUNS = 10

#     random.seed(42)
#     np.random.seed(42)

#     def eval_view_multiple_runs(pairs, view_name: str, cat: str):
#         """Randomly sample up to MAX_PAIRS pairs, repeat N_RUNS times, average metrics."""
#         if len(pairs) == 0:
#             print(f"    [info] No {view_name} pairs for {cat}")
#             return {
#                 "psnr": np.nan,
#                 "ssim": np.nan,
#                 "lpips": np.nan,
#                 "fid": np.nan,
#                 "kid": np.nan,
#             }

#         psnrs, ssims, lpipss = [], [], []
#         fids, kids = [], []

#         for run_idx in range(N_RUNS):
#             if len(pairs) <= MAX_PAIRS:
#                 sampled = pairs
#             else:
#                 sampled = random.sample(pairs, MAX_PAIRS)

#             # pairwise metrics
#             pm = compute_pairwise_metrics(sampled, device=args.device)
#             psnrs.append(pm["psnr"])
#             ssims.append(pm["ssim"])
#             lpipss.append(pm["lpips"])

#             # set-level FID/KID
#             ref_files = [r for (r, g) in sampled]
#             gen_files = [g for (r, g) in sampled]

#             fid = compute_fid_from_lists(
#                 gen_files, ref_files, args.batch_size, args.device, args.num_workers
#             )
#             kid_mean, _ = compute_kid_from_lists(
#                 gen_files, ref_files, args.batch_size, args.device,
#                 args.num_workers, subset_size=args.kid_subset, num_subsets=args.kid_subsets
#             )

#             fids.append(fid)
#             kids.append(kid_mean)

#         psnrs = np.array(psnrs, dtype=np.float64)
#         ssims = np.array(ssims, dtype=np.float64)
#         lpipss = np.array(lpipss, dtype=np.float64)
#         fids = np.array(fids, dtype=np.float64)
#         kids = np.array(kids, dtype=np.float64)

#         return {
#             "psnr": float(psnrs.mean()),
#             "ssim": float(ssims.mean()),
#             "lpips": float(lpipss.mean()),
#             "fid": float(fids.mean()),
#             "kid": float(kids.mean()),
#         }

#     for cat in FARFETCH_CATEGORIES:
#         ref_root = os.path.join(args.ref_root_base, cat) 


#         # ------------------------------------------------------------------
#         # Select generation output folder based on --method
#         # ------------------------------------------------------------------
#         if args.method == 'kiss3dgen':
#             gen_root = os.path.join(args.gen_root_base, cat, "image_to_3d")

#         elif args.method == 'trellis_glb':
#             gen_root = os.path.join(args.gen_root_base, cat)

#         elif args.method == 'Ours':
#             gen_root = os.path.join(args.gen_root_base, cat)

#         elif args.method == 'instantMesh':
#             gen_root = os.path.join(args.gen_root_base, cat, "instant-mesh-large", "meshes")

#         elif args.method == '3Dtopia':
#             gen_root = os.path.join(args.gen_root_base, cat, "inference", "3dtopia-xl-sview")

#         else:
#             raise ValueError(f"Unsupported method: {args.method}")


#         if not os.path.isdir(ref_root):
#             print(f"[skip] Missing ref category folder: {ref_root}")
#             continue
#         if not os.path.isdir(gen_root):
#             print(f"[skip] Missing gen category folder: {gen_root}")
#             continue

#         print(f"\n=== Category: {cat} ===")
#         print(f"ref_root: {ref_root}")
#         print(f"gen_root: {gen_root}")

#         # 1) collect matched pairs from labels.json
#         pairs = collect_pairs_from_json(ref_root, gen_root)
#         front_pairs = pairs["front"]
#         back_pairs  = pairs["back"]

#         print(f"Matched pairs (total): front={len(front_pairs)}  back={len(back_pairs)}")

#         if len(front_pairs) == 0 and len(back_pairs) == 0:
#             print("[warn] No pairs found for this category, skipping.")
#             continue

#         # 2) run repeated random sampling per view
#         print("  [eval] front-view (10× random sampling)")
#         front_metrics = eval_view_multiple_runs(front_pairs, "front", cat) if len(front_pairs) else {
#             "psnr": np.nan, "ssim": np.nan, "lpips": np.nan, "fid": np.nan, "kid": np.nan
#         }

#         print("  [eval] back-view (10× random sampling)")
#         back_metrics = eval_view_multiple_runs(back_pairs, "back", cat) if len(back_pairs) else {
#             "psnr": np.nan, "ssim": np.nan, "lpips": np.nan, "fid": np.nan, "kid": np.nan
#         }

#         # 3) accumulate per-category results for averaging
#         if len(front_pairs):
#             front_psnr_list.append(front_metrics["psnr"])
#             front_ssim_list.append(front_metrics["ssim"])
#             front_lpips_list.append(front_metrics["lpips"])
#             front_fid_list.append(front_metrics["fid"])
#             front_kid_list.append(front_metrics["kid"])

#         if len(back_pairs):
#             back_psnr_list.append(back_metrics["psnr"])
#             back_ssim_list.append(back_metrics["ssim"])
#             back_lpips_list.append(back_metrics["lpips"])
#             back_fid_list.append(back_metrics["fid"])
#             back_kid_list.append(back_metrics["kid"])

#         # 4) store per-category rows for Excel (front first, then back later)
#         if len(front_pairs):
#             rows_front.append({
#                 "category": cat,
#                 "view": "front_view",
#                 "PSNR": front_metrics["psnr"],
#                 "SSIM": front_metrics["ssim"],
#                 "LPIPS": front_metrics["lpips"],
#                 "FID": front_metrics["fid"],
#                 "KID": front_metrics["kid"],
#             })

#         if len(back_pairs):
#             rows_back.append({
#                 "category": cat,
#                 "view": "back_view",
#                 "PSNR": back_metrics["psnr"],
#                 "SSIM": back_metrics["ssim"],
#                 "LPIPS": back_metrics["lpips"],
#                 "FID": back_metrics["fid"],
#                 "KID": back_metrics["kid"],
#             })

#         # optional: print per-category summary
#         print("\n  [Category summary: front (10× avg)]")
#         if len(front_pairs):
#             print(f"    PSNR: {front_metrics['psnr']:.4f}, SSIM: {front_metrics['ssim']:.4f}, "
#                   f"LPIPS: {front_metrics['lpips']:.4f}, FID: {front_metrics['fid']:.4f}, "
#                   f"KID: {front_metrics['kid']:.4f}")
#         else:
#             print("    No front-view pairs.")

#         print("  [Category summary: back (10× avg)]")
#         if len(back_pairs):
#             print(f"    PSNR: {back_metrics['psnr']:.4f}, SSIM: {back_metrics['ssim']:.4f}, "
#                   f"LPIPS: {back_metrics['lpips']:.4f}, FID: {back_metrics['fid']:.4f}, "
#                   f"KID: {back_metrics['kid']:.4f}")
#         else:
#             print("    No back-view pairs.")

#     # 5) average across categories (ignoring NaN)
#     def nanmean(lst):
#         arr = np.array(lst, dtype=np.float64)
#         return float(np.nanmean(arr)) if np.isfinite(arr).any() else float('nan')

#     front_avg_psnr  = nanmean(front_psnr_list)
#     front_avg_ssim  = nanmean(front_ssim_list)
#     front_avg_lpips = nanmean(front_lpips_list)
#     front_avg_fid   = nanmean(front_fid_list)
#     front_avg_kid   = nanmean(front_kid_list)

#     back_avg_psnr  = nanmean(back_psnr_list)
#     back_avg_ssim  = nanmean(back_ssim_list)
#     back_avg_lpips = nanmean(back_lpips_list)
#     back_avg_fid   = nanmean(back_fid_list)
#     back_avg_kid   = nanmean(back_kid_list)

#     print("\n================ Overall averages across categories (10× random sampling, MAX_PAIRS={}) ================".format(MAX_PAIRS))
#     print("Front view:")
#     print(f"  PSNR: {front_avg_psnr:.4f}, SSIM: {front_avg_ssim:.4f}, LPIPS: {front_avg_lpips:.4f}, FID: {front_avg_fid:.4f}, KID: {front_avg_kid:.4f}")
#     print("Back view:")
#     print(f"  PSNR: {back_avg_psnr:.4f}, SSIM: {back_avg_ssim:.4f}, LPIPS: {back_avg_lpips:.4f}, FID: {back_avg_fid:.4f}, KID: {back_avg_kid:.4f}")

#     # 6) average rows for Excel
#     rows_avg = [
#         {
#             "category": "ALL",
#             "view": "front_view",
#             "PSNR": front_avg_psnr,
#             "SSIM": front_avg_ssim,
#             "LPIPS": front_avg_lpips,
#             "FID": front_avg_fid,
#             "KID": front_avg_kid,
#         },
#         {
#             "category": "ALL",
#             "view": "back_view",
#             "PSNR": back_avg_psnr,
#             "SSIM": back_avg_ssim,
#             "LPIPS": back_avg_lpips,
#             "FID": back_avg_fid,
#             "KID": back_avg_kid,
#         },
#     ]

#     # 7) concatenate in desired order: all front, then all back, then averages
#     excel_rows = rows_front + rows_back + rows_avg

#     df = pd.DataFrame(excel_rows, columns=["category", "view", "PSNR", "SSIM", "LPIPS", "FID", "KID"])
#     df.to_excel(args.out_xlsx, index=False)
#     print(f"\n[info] Saved per-category (10× averaged) and overall metrics to: {args.out_xlsx}")

def main_mul_cat_detail_10run():
    parser = argparse.ArgumentParser(
        description="Compute PSNR, SSIM, LPIPS, FID, KID for bag renders vs GT (front/back) over multiple categories with repeated random sampling."
    )
    parser.add_argument(
        '--ref_root_base',
        type=str,
        default='/mnt/data/mdm/datasets/bag/farfetch/women/category_high_labeled',
        help='GT (reference) base root with per-category subfolders.'
    )
    # ['instantMesh','trellis_glb','kiss3dgen','3Dtopia-XL','Ours']
    parser.add_argument('--method', type=str, default='trellis_glb', help='Method name') 
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for Inception activations.')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers for activations.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for metrics that use torch.')
    parser.add_argument('--kid_subset', type=int, default=50, help='Subset size per KID estimate.')
    parser.add_argument('--kid_subsets', type=int, default=10, help='Number of KID subsets to average.')
    parser.add_argument('--out_xlsx', type=str, default='results_150/kiss3dgen.xlsx', help='Output Excel file.')

    # Farfetch categories to evaluate
    FARFETCH_CATEGORIES = [
        "Backpacks",
        "Beach_Bags",
        "Belt_Bags",
        "Bucket_Bags",
        "Clutch_Bags",
        "Luggage",
        "Messenger_&_Crossbody_Bags",
        "Mini_Bags",
        "Shoulder_Bags",
        "Tote_Bags",
    ]

    args = parser.parse_args()

    # define the method root
    # ['instantMesh','trellis_glb','kiss3dgen','3Dtopia-XL','Ours']
    args.gen_root_base = '/mnt/process/mdm/bag/mdm_output/farfetch/women/' + args.method
    args.out_xlsx = 'results_200/' + args.method + '.xlsx'

    # for computing averages over categories (store means only here)
    front_psnr_list, front_ssim_list, front_lpips_list = [], [], []
    front_fid_list, front_kid_list = [], []

    back_psnr_list, back_ssim_list, back_lpips_list = [], [], []
    back_fid_list, back_kid_list = [], []

    # for saving per-category results (ordered as requested)
    rows_front = []
    rows_back = []

    MAX_PAIRS = 200
    N_RUNS = 5

    random.seed(42)
    np.random.seed(42)

    def eval_view_multiple_runs(pairs, view_name: str, cat: str):
        """
        Randomly sample up to MAX_PAIRS pairs, repeat N_RUNS times.
        Return mean and std for each metric over the runs.
        """
        if len(pairs) == 0:
            print(f"    [info] No {view_name} pairs for {cat}")
            return {
                "psnr": np.nan, "psnr_std": np.nan,
                "ssim": np.nan, "ssim_std": np.nan,
                "lpips": np.nan, "lpips_std": np.nan,
                "fid": np.nan, "fid_std": np.nan,
                "kid": np.nan, "kid_std": np.nan,
            }

        psnrs, ssims, lpipss = [], [], []
        fids, kids = [], []

        for run_idx in range(N_RUNS):
            if len(pairs) <= MAX_PAIRS:
                sampled = pairs
            else:
                sampled = random.sample(pairs, MAX_PAIRS)

            # pairwise metrics
            pm = compute_pairwise_metrics(sampled, device=args.device)
            psnrs.append(pm["psnr"])
            ssims.append(pm["ssim"])
            lpipss.append(pm["lpips"])

            # set-level FID/KID
            ref_files = [r for (r, g) in sampled]
            gen_files = [g for (r, g) in sampled]

            fid = compute_fid_from_lists(
                gen_files, ref_files, args.batch_size, args.device, args.num_workers
            )
            kid_mean, _ = compute_kid_from_lists(
                gen_files, ref_files, args.batch_size, args.device,
                args.num_workers, subset_size=args.kid_subset, num_subsets=args.kid_subsets
            )

            fids.append(fid)
            kids.append(kid_mean)

        psnrs = np.array(psnrs, dtype=np.float64)
        ssims = np.array(ssims, dtype=np.float64)
        lpipss = np.array(lpipss, dtype=np.float64)
        fids = np.array(fids, dtype=np.float64)
        kids = np.array(kids, dtype=np.float64)

        return {
            "psnr": float(psnrs.mean()),
            "psnr_std": float(psnrs.std()),
            "ssim": float(ssims.mean()),
            "ssim_std": float(ssims.std()),
            "lpips": float(lpipss.mean()),
            "lpips_std": float(lpipss.std()),
            "fid": float(fids.mean()),
            "fid_std": float(fids.std()),
            "kid": float(kids.mean()),
            "kid_std": float(kids.std()),
        }

    for cat in FARFETCH_CATEGORIES:
        ref_root = os.path.join(args.ref_root_base, cat) 

        # ------------------------------------------------------------------
        # Select generation output folder based on --method
        # ------------------------------------------------------------------
        if args.method == 'kiss3dgen':
            gen_root = os.path.join(args.gen_root_base, cat, "image_to_3d")

        elif args.method == 'trellis_glb':
            gen_root = os.path.join(args.gen_root_base, cat)

        elif args.method == 'Ours':
            gen_root = os.path.join(args.gen_root_base, cat)

        elif args.method == 'instantMesh':
            gen_root = os.path.join(args.gen_root_base, cat, "instant-mesh-large", "meshes")

        elif args.method == '3Dtopia-XL':
            gen_root = os.path.join(args.gen_root_base, cat, "inference", "3dtopia-xl-sview")

        else:
            raise ValueError(f"Unsupported method: {args.method}")

        if not os.path.isdir(ref_root):
            print(f"[skip] Missing ref category folder: {ref_root}")
            continue
        if not os.path.isdir(gen_root):
            print(f"[skip] Missing gen category folder: {gen_root}")
            continue

        print(f"\n=== Category: {cat} ===")
        print(f"ref_root: {ref_root}")
        print(f"gen_root: {gen_root}")

        # 1) collect matched pairs from labels.json
        pairs = collect_pairs_from_json(ref_root, gen_root)
        front_pairs = pairs["front"]
        back_pairs  = pairs["back"]

        print(f"Matched pairs (total): front={len(front_pairs)}  back={len(back_pairs)}")

        if len(front_pairs) == 0 and len(back_pairs) == 0:
            print("[warn] No pairs found for this category, skipping.")
            continue

        # 2) run repeated random sampling per view
        print("  [eval] front-view (10× random sampling)")
        front_metrics = eval_view_multiple_runs(front_pairs, "front", cat)

        print("  [eval] back-view (10× random sampling)")
        back_metrics = eval_view_multiple_runs(back_pairs, "back", cat)

        # 3) accumulate per-category results for averaging (using means)
        if len(front_pairs):
            front_psnr_list.append(front_metrics["psnr"])
            front_ssim_list.append(front_metrics["ssim"])
            front_lpips_list.append(front_metrics["lpips"])
            front_fid_list.append(front_metrics["fid"])
            front_kid_list.append(front_metrics["kid"])

        if len(back_pairs):
            back_psnr_list.append(back_metrics["psnr"])
            back_ssim_list.append(back_metrics["ssim"])
            back_lpips_list.append(back_metrics["lpips"])
            back_fid_list.append(back_metrics["fid"])
            back_kid_list.append(back_metrics["kid"])

        # 4) store per-category rows for Excel (front first, then back later)
        if len(front_pairs):
            rows_front.append({
                "category": cat,
                "view": "front_view",
                "PSNR": front_metrics["psnr"],
                "PSNR_std": front_metrics["psnr_std"],
                "SSIM": front_metrics["ssim"],
                "SSIM_std": front_metrics["ssim_std"],
                "LPIPS": front_metrics["lpips"],
                "LPIPS_std": front_metrics["lpips_std"],
                "FID": front_metrics["fid"],
                "FID_std": front_metrics["fid_std"],
                "KID": front_metrics["kid"],
                "KID_std": front_metrics["kid_std"],
            })

        if len(back_pairs):
            rows_back.append({
                "category": cat,
                "view": "back_view",
                "PSNR": back_metrics["psnr"],
                "PSNR_std": back_metrics["psnr_std"],
                "SSIM": back_metrics["ssim"],
                "SSIM_std": back_metrics["ssim_std"],
                "LPIPS": back_metrics["lpips"],
                "LPIPS_std": back_metrics["lpips_std"],
                "FID": back_metrics["fid"],
                "FID_std": back_metrics["fid_std"],
                "KID": back_metrics["kid"],
                "KID_std": back_metrics["kid_std"],
            })

        # optional: print per-category summary
        print("\n  [Category summary: front (10× avg ± std)]")
        if len(front_pairs):
            print(f"    PSNR: {front_metrics['psnr']:.4f} ± {front_metrics['psnr_std']:.4f}, "
                  f"SSIM: {front_metrics['ssim']:.4f} ± {front_metrics['ssim_std']:.4f}, "
                  f"LPIPS: {front_metrics['lpips']:.4f} ± {front_metrics['lpips_std']:.4f}, "
                  f"FID: {front_metrics['fid']:.4f} ± {front_metrics['fid_std']:.4f}, "
                  f"KID: {front_metrics['kid']:.4f} ± {front_metrics['kid_std']:.4f}")
        else:
            print("    No front-view pairs.")

        print("  [Category summary: back (10× avg ± std)]")
        if len(back_pairs):
            print(f"    PSNR: {back_metrics['psnr']:.4f} ± {back_metrics['psnr_std']:.4f}, "
                  f"SSIM: {back_metrics['ssim']:.4f} ± {back_metrics['ssim_std']:.4f}, "
                  f"LPIPS: {back_metrics['lpips']:.4f} ± {back_metrics['lpips_std']:.4f}, "
                  f"FID: {back_metrics['fid']:.4f} ± {back_metrics['fid_std']:.4f}, "
                  f"KID: {back_metrics['kid']:.4f} ± {back_metrics['kid_std']:.4f}")
        else:
            print("    No back-view pairs.")

    # 5) average across categories (ignoring NaN) – with std
    def nanmean_std(lst):
        arr = np.array(lst, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return float('nan'), float('nan')
        return float(arr.mean()), float(arr.std())

    front_avg_psnr,  front_std_psnr  = nanmean_std(front_psnr_list)
    front_avg_ssim,  front_std_ssim  = nanmean_std(front_ssim_list)
    front_avg_lpips, front_std_lpips = nanmean_std(front_lpips_list)
    front_avg_fid,   front_std_fid   = nanmean_std(front_fid_list)
    front_avg_kid,   front_std_kid   = nanmean_std(front_kid_list)

    back_avg_psnr,  back_std_psnr  = nanmean_std(back_psnr_list)
    back_avg_ssim,  back_std_ssim  = nanmean_std(back_ssim_list)
    back_avg_lpips, back_std_lpips = nanmean_std(back_lpips_list)
    back_avg_fid,   back_std_fid   = nanmean_std(back_fid_list)
    back_avg_kid,   back_std_kid   = nanmean_std(back_kid_list)

    print("\n================ Overall averages across categories (10× random sampling, MAX_PAIRS={}) ================".format(MAX_PAIRS))
    print("Front view:")
    print(f"  PSNR: {front_avg_psnr:.4f} ± {front_std_psnr:.4f}, "
          f"SSIM: {front_avg_ssim:.4f} ± {front_std_ssim:.4f}, "
          f"LPIPS: {front_avg_lpips:.4f} ± {front_std_lpips:.4f}, "
          f"FID: {front_avg_fid:.4f} ± {front_std_fid:.4f}, "
          f"KID: {front_avg_kid:.4f} ± {front_std_kid:.4f}")
    print("Back view:")
    print(f"  PSNR: {back_avg_psnr:.4f} ± {back_std_psnr:.4f}, "
          f"SSIM: {back_avg_ssim:.4f} ± {back_std_ssim:.4f}, "
          f"LPIPS: {back_avg_lpips:.4f} ± {back_std_lpips:.4f}, "
          f"FID: {back_avg_fid:.4f} ± {back_std_fid:.4f}, "
          f"KID: {back_avg_kid:.4f} ± {back_std_kid:.4f}")

    # 6) average rows for Excel
    rows_avg = [
        {
            "category": "ALL",
            "view": "front_view",
            "PSNR": front_avg_psnr,
            "PSNR_std": front_std_psnr,
            "SSIM": front_avg_ssim,
            "SSIM_std": front_std_ssim,
            "LPIPS": front_avg_lpips,
            "LPIPS_std": front_std_lpips,
            "FID": front_avg_fid,
            "FID_std": front_std_fid,
            "KID": front_avg_kid,
            "KID_std": front_std_kid,
        },
        {
            "category": "ALL",
            "view": "back_view",
            "PSNR": back_avg_psnr,
            "PSNR_std": back_std_psnr,
            "SSIM": back_avg_ssim,
            "SSIM_std": back_std_ssim,
            "LPIPS": back_avg_lpips,
            "LPIPS_std": back_std_lpips,
            "FID": back_avg_fid,
            "FID_std": back_std_fid,
            "KID": back_avg_kid,
            "KID_std": back_std_kid,
        },
    ]

    # 7) concatenate in desired order: all front, then all back, then averages
    excel_rows = rows_front + rows_back + rows_avg

    df = pd.DataFrame(
        excel_rows,
        columns=[
            "category", "view",
            "PSNR", "PSNR_std",
            "SSIM", "SSIM_std",
            "LPIPS", "LPIPS_std",
            "FID", "FID_std",
            "KID", "KID_std",
        ]
    )
    df.to_excel(args.out_xlsx, index=False)
    print(f"\n[info] Saved per-category (10× mean ± std) and overall metrics to: {args.out_xlsx}")


if __name__ == '__main__':
    # main() ##compute based on a single category
    # main_mul_cat() ## compuate the average based on all category, save average results
    # main_mul_cat_detail() ## compuate the average based on all category, save detailed results
    main_mul_cat_detail_10run() ## compuate the average based on all category, save detailed results
