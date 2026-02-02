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
    return float(vals.mean()), float(vals.std())


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

if __name__ == '__main__':
    main()
