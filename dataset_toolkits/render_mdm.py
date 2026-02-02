import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import call
import numpy as np
from utils import sphere_hammersley_sequence

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def _render(file_path, sha256, output_dir, num_views, script_path):
    output_folder = os.path.join(output_dir, 'renders', sha256)
    os.makedirs(output_folder, exist_ok=True)

    # Build camera {yaw, pitch, radius, fov}
    yaws, pitchs = [], []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y); pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    args = [
        BLENDER_PATH, '-b', '-P', script_path,
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)

    # Per-object log (so failures are visible)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{sha256}.log")
    with open(log_path, "w") as lf:
        ret = call(args, stdout=lf, stderr=lf)

    # Success criteria: transforms.json exists (as your script expects)
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'sha256': sha256, 'rendered': True}
    else:
        # Drop a small marker for quick grep
        with open(os.path.join(log_dir, f"{sha256}.failed"), "w") as ff:
            ff.write(f"Return code: {ret}\nArgs: {args}\n")
        return {'sha256': sha256, 'rendered': False}

if __name__ == '__main__':
    # dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')
    dataset_utils = importlib.import_module("datasets.ABO")  # fixed to ABO

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="/mnt/data/mdm/datasets/ABO",
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=2)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)

    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # Sanity: blender binary and blender_script exist
    if not (os.path.exists(BLENDER_PATH) and os.access(BLENDER_PATH, os.X_OK)):
        raise FileNotFoundError(f"Blender not found or not executable at {BLENDER_PATH}")
    script_path = os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py')
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Blender script missing: {script_path}")

    # get file list
    meta_path = os.path.join(opt.output_dir, 'metadata.csv')
    if not os.path.exists(meta_path):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(meta_path)

    # --- Normalize local_path to absolute path and filter to existing files ---
    if 'local_path' not in metadata.columns:
        raise KeyError("Expected 'local_path' column is missing from metadata.csv")

    # make absolute if relative
    def to_abs(p):
        p = str(p)
        return p if os.path.isabs(p) else os.path.join(opt.output_dir, p)

    metadata['local_path'] = metadata['local_path'].astype(str).map(to_abs)
    metadata = metadata[metadata['local_path'].notna() & metadata['local_path'].map(os.path.exists)]

    # optional: aesthetic score filter
    if opt.filter_low_aesthetic_score is not None and 'aesthetic_score' in metadata.columns:
        metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]

    # optional: skip already rendered (based on column)
    if 'rendered' in metadata.columns:
        metadata = metadata[metadata['rendered'] == False]

    # instance subset
    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # split by rank/world_size
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata.iloc[start:end].copy()

    # filter out objects that are already processed on disk
    records = []
    for sha256 in list(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]

    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_render, output_dir=opt.output_dir, num_views=opt.num_views, script_path=script_path)
    rendered = dataset_utils.foreach_instance(
        metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects'
    )
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)], ignore_index=True)
    rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)
