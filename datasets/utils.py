import torch
from torchvision import transforms
from torch.utils.data import Dataset, Sampler
import numpy as np
from shapely.geometry import Polygon
from PIL import Image
import math

# CLASSES FOR MULTI-DATASET SAMPLING

class MultiDatasetSampler(Sampler):
    def __init__(self, datasets, probs, num_samples=None):
        """
        datasets: List of dataset instances
        probs: List of probabilities for each dataset
        num_samples: Total number of samples to draw
        """
        self.datasets = datasets
        self.probs = np.array(probs) / sum(probs)  # Normalize to sum to 1

        # Compute the cumulative size of datasets
        self.dataset_sizes = [len(d) for d in datasets]
        self.dataset_indices = [list(range(s)) for s in self.dataset_sizes]
        print(self.dataset_sizes)
        self.num_samples = min(self.dataset_sizes) if num_samples is None else num_samples
        print(self.num_samples)

    def __iter__(self):
        sampled_dataset_indices = np.random.choice(
            len(self.datasets),
            size=self.num_samples,
            p=self.probs,
        )
        # At init time — create mutable pools of available indices
        self.available_indices = {
            i: list(self.dataset_indices[i]) for i in range(len(self.datasets))
        }

        final_indices = []
        for dataset_idx in sampled_dataset_indices:
            if not self.available_indices[dataset_idx]:
                # If the pool is empty, refill it
                self.refill()
            # Randomly pick one index from available pool
            chosen_idx = np.random.choice(self.available_indices[dataset_idx])
            # Remove the index from the pool to ensure no replacement
            self.available_indices[dataset_idx].remove(chosen_idx)   
            final_indices.append((dataset_idx, chosen_idx))

        return iter(final_indices)
    def refill(self):
        # Refill the available indices for each dataset
        self.available_indices = {
            i: list(self.dataset_indices[i]) for i in range(len(self.datasets))
        }
    def __len__(self):
        return self.num_samples

class MultiDatasetWrapper(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    @property
    def is_synthetic(self):
        return any([dataset.is_synthetic for dataset in self.datasets])

    def __getitem__(self, index):
        dataset_idx, sample_idx = index
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return sum(len(d) for d in self.datasets)

# HELPERS FOR POLYGONS

def get_polygons_from_json(data, n_vertices=None):
    polygons = []
    for shape in data['shapes']:
        polygons.append(torch.tensor(shape['points']))
    if n_vertices == None:
        return polygons
    else:
        return interpolate_vertices(polygons, N=n_vertices+1)

def polygon_perimeter(polygon):
    """Calculate the perimeter of the polygon (sum of edge lengths)."""
    perimeter = 0
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]  # Wrap around to the first vertex
        perimeter += np.linalg.norm(p2 - p1)
    return perimeter

def resample_polygon(polygon, num_vertices):
    """Resample the polygon by placing vertices equally spaced along the perimeter."""
    perimeter = polygon_perimeter(polygon)
    # Calculate the desired distance between consecutive vertices
    segment_length = perimeter / (num_vertices - 1)

    # List to store the resampled points
    resampled = [polygon[0]]  # Start with the first point
    current_length = 0  # Keep track of the length along the polygon's perimeter

    # Traverse the edges of the polygon
    for i in range(0, len(polygon)):
        p1 = polygon[i % len(polygon)]  # Current vertex
        p2 = polygon[(i + 1) % len(polygon)]  # Next vertex (wrap around)
        edge_length = np.linalg.norm(p2 - p1)

        # While the remaining length in the current edge can fit another vertex
        while current_length + edge_length > segment_length:
            # Calculate the proportion along the edge where the new point should be
            excess_length = segment_length - current_length
            ratio = excess_length / edge_length
            new_point = p1 + ratio * (p2 - p1)
            resampled.append(new_point)

            # Update the remaining length to be 0 (start a new segment)
            current_length = 0
            edge_length -= excess_length
            p1 = new_point  # Start a new edge from the last point

        # If we haven't yet added the vertex for the current edge, move to the next
        current_length += edge_length
    if len(resampled) != num_vertices:
        resampled.append(resampled[0])
    assert len(resampled) == num_vertices
    return torch.cat(resampled)

def interpolate_vertices(polygons, N):
    """Process multiple polygons to resample them to the desired number of vertices."""
    resampled_polygons = []
    for polygon in polygons:
        if polygon.shape[0] == N:
            resampled_polygon = polygon.view(N*2)
        elif polygon.shape[0] < 3:
            continue
        elif not Polygon(polygon).is_valid:
            continue
        else:
            resampled_polygon = resample_polygon(polygon, N).view(N,2)[:-1,:].view((N-1)*2)
        resampled_polygons.append(resampled_polygon)
    return resampled_polygons

def is_valid(batch_polygons):
    """
    Vectorized check if each polygon in a batch is valid (i.e., has area > 0 and is not self-intersecting).
    
    Args:
        batch_polygons (np.ndarray): A (B, num_points * 2) array representing B polygons,
                                     where each polygon has (x1, y1, x2, y2, ..., xn, yn).

    Returns:
        np.ndarray: A (B,) boolean array indicating whether each polygon is valid.
    """
    B, num_coords = batch_polygons.shape
    num_points = num_coords // 2  # Each point has (x, y)

    # Reshape to (B, num_points, 2) where each row is a list of (x, y) pairs
    polygons = batch_polygons.reshape(B, num_points, 2)

    # Convert to Shapely Polygons in a vectorized way using map
    shapely_polygons = np.array([Polygon(p) for p in polygons])

    # Extract validity and area efficiently
    valid_mask = np.array([p.is_valid and p.area > 0 for p in shapely_polygons])

    return valid_mask

def rotate_point(x, y, cx, cy, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    nx = cos_a * (x - cx) - sin_a * (y - cy) + cx
    ny = sin_a * (x - cx) + cos_a * (y - cy) + cy
    return nx, ny

# HELPERS FOR STYLE TRANSFER

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def _style_transfer(vgg, decoder, device, pil_content, pil_style, alpha=0.1): 
    content_size = 512
    style_size = 512
    crop = False
    
    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    
    content = content_tf(pil_content)
    style = style_tf(pil_style)

    content = content.to(device).unsqueeze(0)
    style = style.to(device).unsqueeze(0)

    with torch.no_grad():
        assert (0.0 <= alpha <= 1.0)
        content_f = vgg(content)
        style_f = vgg(style)
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        output = decoder(feat)
    output = output.cpu()
    output = torch.clamp(output, min=0, max=1)
    return transforms.ToPILImage()(output[0])

def style_transfer(I, heavy_loads, style_path):
    ## load style transfer network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = heavy_loads['decoder'].eval()
    vgg_ = heavy_loads['vgg'].eval()
    style_pil = Image.open(style_path).convert('RGB')
    I = I.convert('RGB')
    I_stylised = _style_transfer(vgg_, decoder, device, I, style_pil, alpha=0.5)    
    I_stylised = I_stylised.resize(I.size, resample=2)
    return I_stylised

# UTIL DINO-DETR

import os
import shutil
import time
import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import slconfig

class Error(OSError):
    pass

def slcopytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copyfile,
             ignore_dangling_symlinks=False):
    """
    modified from shutil.copytree without copystat.
    
    Recursively copy a directory tree.

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied. If the file pointed by the symlink doesn't
    exist, an exception will be added in the list of errors raised in
    an Error exception at the end of the copy process.

    You can set the optional ignore_dangling_symlinks flag to true if you
    want to silence this exception. Notice that this has no effect on
    platforms that don't support os.symlink.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    The optional copy_function argument is a callable that will be used
    to copy each file. It will be called with the source path and the
    destination path as arguments. By default, copy2() is used, but any
    function that supports the same signature (like copy()) can be used.

    """
    errors = []
    if os.path.isdir(src):
        names = os.listdir(src)
        if ignore is not None:
            ignored_names = ignore(src, names)
        else:
            ignored_names = set()

        os.makedirs(dst)
        for name in names:
            if name in ignored_names:
                continue
            srcname = os.path.join(src, name)
            dstname = os.path.join(dst, name)
            try:
                if os.path.islink(srcname):
                    linkto = os.readlink(srcname)
                    if symlinks:
                        # We can't just leave it to `copy_function` because legacy
                        # code with a custom `copy_function` may rely on copytree
                        # doing the right thing.
                        os.symlink(linkto, dstname)
                    else:
                        # ignore dangling symlink if the flag is on
                        if not os.path.exists(linkto) and ignore_dangling_symlinks:
                            continue
                        # otherwise let the copy occurs. copy2 will raise an error
                        if os.path.isdir(srcname):
                            slcopytree(srcname, dstname, symlinks, ignore,
                                    copy_function)
                        else:
                            copy_function(srcname, dstname)
                elif os.path.isdir(srcname):
                    slcopytree(srcname, dstname, symlinks, ignore, copy_function)
                else:
                    # Will raise a SpecialFileError for unsupported file types
                    copy_function(srcname, dstname)
            # catch the Error from the recursive copytree so that we can
            # continue with other files
            except Error as err:
                errors.extend(err.args[0])
            except OSError as why:
                errors.append((srcname, dstname, str(why)))
    else:
        copy_function(src, dst)

    if errors:
        raise Error(errors)
    return dst

def check_and_copy(src_path, tgt_path):
    if os.path.exists(tgt_path):
        return None

    return slcopytree(src_path, tgt_path)

def remove(srcpath):
    if os.path.isdir(srcpath):
        return shutil.rmtree(srcpath)
    else:
        return os.remove(srcpath)  

def preparing_dataset(pathdict, image_set, args):
    start_time = time.time()
    dataset_file = args.dataset_file
    data_static_info = slconfig.SLConfig.fromfile('util/static_data_path.py')
    static_dict = data_static_info[dataset_file][image_set]

    copyfilelist = []
    for k,tgt_v in pathdict.items():
        if os.path.exists(tgt_v):
            if args.local_rank == 0:
                print("path <{}> exist. remove it!".format(tgt_v))
                remove(tgt_v)
            # continue
        
        if args.local_rank == 0:
            src_v = static_dict[k]
            assert isinstance(src_v, str)
            if src_v.endswith('.zip'):
                # copy
                cp_tgt_dir = os.path.dirname(tgt_v)
                filename = os.path.basename(src_v)
                cp_tgt_path = os.path.join(cp_tgt_dir, filename)
                print('Copy from <{}> to <{}>.'.format(src_v, cp_tgt_path))
                os.makedirs(cp_tgt_dir, exist_ok=True)
                check_and_copy(src_v, cp_tgt_path)          

                # unzip
                import zipfile
                print("Starting unzip <{}>".format(cp_tgt_path))
                with zipfile.ZipFile(cp_tgt_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(cp_tgt_path))      

                copyfilelist.append(cp_tgt_path)
                copyfilelist.append(tgt_v)
            else:
                print('Copy from <{}> to <{}>.'.format(src_v, tgt_v))
                os.makedirs(os.path.dirname(tgt_v), exist_ok=True)
                check_and_copy(src_v, tgt_v)
                copyfilelist.append(tgt_v)
    
    if len(copyfilelist) == 0:
        copyfilelist = None
    args.copyfilelist = copyfilelist
        
    if args.distributed:
        torch.distributed.barrier()
    total_time = time.time() - start_time
    if copyfilelist:
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Data copy time {}'.format(total_time_str))
    return copyfilelist