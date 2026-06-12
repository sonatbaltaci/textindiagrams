if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))
import torch
import os
import pickle
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import datasets.transforms as T
import xml.etree.ElementTree as ET
from scipy.interpolate import CubicSpline

current_dir = os.path.dirname(os.path.abspath(__file__))

if 'dataset' not in current_dir:
    current_dir = os.path.join(current_dir, 'dataset')
else:
    current_dir = os.path.join(current_dir, '.')

#with open(os.path.join(current_dir, 'config.json'), 'r') as f:
datasets_path = '/home/rbaena/datasets/'
import torch
from torch import Tensor

def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated values, shape `(batch_size, *)`.
        xp: the :math:`x`-coordinates of the data points, must be increasing, shape `(n,)`.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`, shape `(n,)`.

    Returns:
        the interpolated values, same size as `x`.
    """
    batch_size = x.shape[0]
    device = x.device

    # Reshape xp and fp to broadcast with x
    xp = xp.unsqueeze(0).repeat(batch_size, 1).to(device)
    fp = fp.unsqueeze(0).repeat(batch_size, 1).to(device)

    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    # Compute indices for each sample in the batch
    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 2) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    # Gather the corresponding values of m and b for each sample
    m_gather = torch.gather(m, 1, indicies)
    b_gather = torch.gather(b, 1, indicies)

    return m_gather * x + b_gather
def interpolate_polygones(list_polygons,num_points=5):
    interpolated_polygones =[]
    for pp in list_polygons:
        pp = torch.tensor(pp).flatten()
        nb_pts = pp.shape[0]
        bottom_polygones = pp[:nb_pts//2]
        x_pts_bottom = bottom_polygones[::2]
        y_pts_bottom = bottom_polygones[1::2]
        x_pts_range_bottom = np.linspace(x_pts_bottom[0],x_pts_bottom[-1],num_points)
        x_pts_bottom_sorted, idx_xpts_bottom = x_pts_bottom.sort()
        y_pts_bottom_sorted = y_pts_bottom[idx_xpts_bottom]
        if x_pts_range_bottom[0] != x_pts_range_bottom[-1]:
            y_pts_range_bottom = np.interp(x_pts_range_bottom,x_pts_bottom_sorted,y_pts_bottom_sorted)
        else:
            y_pts_range_bottom = np.linspace(y_pts_bottom[0],y_pts_bottom[-1],num_points)
        x_pts_range_bottom,y_pts_range_bottom = torch.tensor(x_pts_range_bottom),torch.tensor(y_pts_range_bottom)
        bottom_polygones = torch.cat((x_pts_range_bottom.unsqueeze(0),y_pts_range_bottom.unsqueeze(0)),0).T.reshape(-1)

        # is_vertical = False
        # if np.abs(y_pts_bottom[0] - y_pts_bottom[-1]) > np.abs(x_pts_bottom[0] - x_pts_bottom[-1]):
        #     is_vertical = True

        top_polygones = pp[nb_pts//2:]
        x_pts_top = top_polygones[::2]
        y_pts_top = top_polygones[1::2]
        x_pts_range_top = np.linspace(x_pts_top[0],x_pts_top[-1],num_points)
        if x_pts_range_top[0] != x_pts_range_top[-1]:
            x_pts_top_sorted, idx_xpts_top = x_pts_top.sort()
            y_pts_top_sorted = y_pts_top[idx_xpts_top]
            y_pts_range_top = np.interp(x_pts_range_top,x_pts_top_sorted,y_pts_top_sorted)
            x_pts_range_top,y_pts_range_top = torch.tensor(x_pts_range_top),torch.tensor(y_pts_range_top)

        else:
            y_pts_range_top = np.linspace(y_pts_top[0],y_pts_top[-1],num_points)
            x_pts_range_top = torch.tensor(x_pts_range_top)
            y_pts_range_top = torch.tensor(y_pts_range_top)

        top_polygones = torch.cat((x_pts_range_top.unsqueeze(0),y_pts_range_top.unsqueeze(0)),0).T.reshape(-1)
        final_polygone = torch.cat((bottom_polygones,top_polygones),0)
        final_polygone.reshape(-1)

        pp = final_polygone
        interpolated_polygones.append(final_polygone)
    return interpolated_polygones

def interpolate_top_line(top_line, nb_points):
    distances = np.cumsum(np.sqrt(np.sum(np.diff(top_line, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    cs_x = CubicSpline(distances, top_line[:, 0], bc_type='natural')
    cs_y = CubicSpline(distances, top_line[:, 1], bc_type='natural')
    num_samples = nb_points
    uniform_distances = np.linspace(0, distances[-1], num_samples)
    interp_x = cs_x(uniform_distances)
    interp_y = cs_y(uniform_distances)
    return list(zip(interp_x, interp_y))


def read_coordinates_from_xtml(path_label):

    tree = ET.parse(path_label)
    root = tree.getroot()
    ns = {'ns': 'https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    baseline_elements = root.findall(".//ns:Baseline", ns)

    # List to store baselines (each baseline is a list of (x, y) tuples)
    baselines = []
    for elem in baseline_elements:
        points_str = elem.attrib.get("points")
        if points_str:
            # Points are given as "x1,y1 x2,y2 ..." so split by whitespace first.
            points = [tuple(map(int, point.split(','))) for point in points_str.split()]
            baselines.append(points)
    textline_elements = root.findall(".//ns:TextLine", ns)

    # List to store coordinates for each TextLine (each is a list of (x, y) tuples)
    textline_coords = []

    # Loop through each TextLine and extract its <Coords> element
    for textline in textline_elements:
        coords_elem = textline.find("ns:Coords", ns)
        if coords_elem is not None:
            points_str = coords_elem.attrib.get("points")
            if points_str:
                # Points are provided as "x1,y1 x2,y2 ..." so split on whitespace and then by comma.
                points = [tuple(map(int, point.split(','))) for point in points_str.split()]
                textline_coords.append(points)
    return baselines, textline_coords


class cbdar2019(Dataset):
    def __init__(self, mode, transform=transforms.ToTensor(), target_transform=None):
        """ 
        mode: train, valid, test
        """
        if mode =="val":
            mode ="eval"

        self.mode = mode
        self._transforms = transform
        ### load labels (text) from pickle file
        folder_data = os.path.join(datasets_path, 'READ-ICDAR2019-cBAD-dataset', mode)
        self.folder_data = folder_data

        self.data = os.listdir(folder_data)
        print(len(self.data))
        #end jpg
        self.data = [{"idx": os.path.splitext(f)[0]} for f in self.data if f.endswith('.jpg')]
        self.transform = transform
        self.target_transform = target_transform


    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        data_item = self.data[idx]
        #img = Image.open(datasets_path + '/cbdar2019/' + self.mode + '/' +data_item["idx"] + '.jpg')
        img_path = os.path.join(datasets_path, 'READ-ICDAR2019-cBAD-dataset', self.mode, data_item["idx"] + '.jpg')
        xml_path = os.path.join(datasets_path, 'READ-ICDAR2019-cBAD-dataset', self.mode,'page', data_item["idx"] + '.xml')

        image = Image.open(os.path.join(datasets_path, 'READ-ICDAR2019-cBAD-dataset', self.mode, data_item["idx"] + '.jpg'))
        baselines, textline_coords = read_coordinates_from_xtml(xml_path)
        labels = {}
        polygones = []
        for jj, tt in enumerate(textline_coords):
            half_= len(tt)//2
            if len(tt) % 2 == 1 and half_ ==1:
                top_line = top_line[:2]
            else:
                top_line=tt[:half_]
            if jj > len(baselines)-1:
                continue
            baseline = baselines[jj]
            if len(baseline) != len(top_line):
                print('error')
                top_line = interpolate_top_line(np.array(top_line), len(baseline))
            tt = np.concatenate((baseline, top_line[::-1]) )
            tt = interpolate_polygones([tt], num_points = 5)[0]# TO FIX
            # tt =tt[:len(tt)//2]# only the bottom line


            # tt = tt
            polygones.append(tt)
        if len(polygones) == 0:
            polygones = target_boxes = torch.empty((0, 20))
        elif len(polygones) >1:
            polygones = torch.stack(polygones)
        else:
            polygones = torch.tensor(polygones[0]).unsqueeze(0)



        labels["labels"] =  torch.zeros((polygones.shape[0],), dtype=torch.int64)
        labels["orig_size"]  = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
        labels["size"] = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
        labels["img_idx"] = torch.tensor([idx], dtype=torch.int64)
        labels["idx"] = torch.tensor([idx], dtype=torch.int64)

        labels["boxes"] =  polygones.float()

        image, labels = self._transforms(image, labels)

        return image, labels


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # config the params for data aug
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        max_size = 1333
        scales2_resize = [400, 500, 600]
        scales2_crop = [384, 600]
        
        # update args from config files
        scales = getattr(args, 'data_aug_scales', scales)
        max_size = getattr(args, 'data_aug_max_size', max_size)
        scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
        scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

        # resize them
        data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
        if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
            data_aug_scale_overlap = float(data_aug_scale_overlap)
            scales = [int(i*data_aug_scale_overlap) for i in scales]
            max_size = int(max_size*data_aug_scale_overlap)
            scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
            scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

        datadict_for_print = {
            'scales': scales,
            'max_size': max_size,
            'scales2_resize': scales2_resize,
            'scales2_crop': scales2_crop
        }

        if image_set == 'train':
            if fix_size:
                return T.Compose([
                    # T.RandomHorizontalFlip(),
                    T.RandomResize([(max_size, max(scales))]),
                    normalize,
                ])

            if strong_aug:
                import datasets.sltransform as SLT

                return T.Compose([
                    T.RandomSelect(
                        T.RandomResize(scales, max_size=max_size),
                        T.Compose([
                            #T.RandomResize(scales2_resize),
                            #T.RandomSizeCrop(*scales2_crop),
                            T.RandomResize(scales, max_size=max_size),
                        ])
                    ),
                    normalize,
                ])

            return T.Compose([
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        #T.RandomResize(scales2_resize),
                        #T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                normalize,
                # T.blur,
                
            ])

        if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

            if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
                print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
                return T.Compose([
                    T.ResizeDebug((1280, 800)),
                    normalize,
                ])   

            return T.Compose([
                T.RandomResize([max(scales)], max_size=max_size),
                normalize,

            ])



        raise ValueError(f'unknown {image_set}')

def build_cbdar2019(image_set, args):
    transforms = make_coco_transforms(image_set, args=args)
    return cbdar2019(image_set, transforms)
