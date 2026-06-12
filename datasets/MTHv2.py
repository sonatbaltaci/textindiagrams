
if __name__ == "__main__":
    import os, sys
    sys.path.append('../')
import torch
import os
import pickle
import numpy as np
import json
from PIL import Image
from torchvision import transforms
import datasets.transforms as T
from torch.utils.data import Dataset
current_directory = os.getcwd()
print('curr',current_directory)


def polygon_center(points: np.ndarray) -> np.ndarray:
    """
    Compute the centroid (barycenter) of a set of 2D points.
    :param points: N x 2 NumPy array of point coordinates.
    :return: 1 x 2 NumPy array representing the centroid (cx, cy).
    """
    # The mean of each column gives (cx, cy)
    return np.mean(points, axis=0)

def sort_points_by_angle(points: np.ndarray) -> np.ndarray:
    """
    Sort the given 2D points by their polar angle relative to the centroid.
    :param points: N x 2 NumPy array of point coordinates (x, y).
    :return: N x 2 NumPy array of points sorted by angle (counterclockwise).
    """
    center = polygon_center(points)
    
    # Translate points so that the centroid is the origin
    deltas = points - center
    
    # Compute the angle for each point
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    
    # Sort points by ascending angle
    sort_indices = np.argsort(angles)
    sorted_points = points[sort_indices]
    sorted_points = np.roll(sorted_points, -2)
    return sorted_points



def interpolate_polygones(list_polygons):
    interpolated_polygones =[]
    for pp in list_polygons:
        pp = torch.tensor(pp).flatten()
        nb_pts = pp.shape[0]
        bottom_polygones = pp[:nb_pts//2]
        x_pts_bottom = bottom_polygones[::2]
        y_pts_bottom = bottom_polygones[1::2]
        x_pts_range_bottom = np.linspace(x_pts_bottom[0],x_pts_bottom[-1],5)
        x_pts_bottom_sorted, idx_xpts_bottom = x_pts_bottom.sort()
        y_pts_bottom_sorted = y_pts_bottom[idx_xpts_bottom]
        if x_pts_range_bottom[0] != x_pts_range_bottom[-1]:
            y_pts_range_bottom = np.interp(x_pts_range_bottom,x_pts_bottom_sorted,y_pts_bottom_sorted)
        else:
            y_pts_range_bottom = np.linspace(y_pts_bottom[0],y_pts_bottom[-1],5)
        x_pts_range_bottom,y_pts_range_bottom = torch.tensor(x_pts_range_bottom),torch.tensor(y_pts_range_bottom)
        bottom_polygones = torch.cat((x_pts_range_bottom.unsqueeze(0),y_pts_range_bottom.unsqueeze(0)),0).T.reshape(-1)

        # is_vertical = False
        # if np.abs(y_pts_bottom[0] - y_pts_bottom[-1]) > np.abs(x_pts_bottom[0] - x_pts_bottom[-1]):
        #     is_vertical = True

        top_polygones = pp[nb_pts//2:]
        x_pts_top = top_polygones[::2]
        y_pts_top = top_polygones[1::2]
        x_pts_range_top = np.linspace(x_pts_top[0],x_pts_top[-1],5)
        if x_pts_range_top[0] != x_pts_range_top[-1]:
            x_pts_top_sorted, idx_xpts_top = x_pts_top.sort()
            y_pts_top_sorted = y_pts_top[idx_xpts_top]
            y_pts_range_top = np.interp(x_pts_range_top,x_pts_top_sorted,y_pts_top_sorted)
            x_pts_range_top,y_pts_range_top = torch.tensor(x_pts_range_top),torch.tensor(y_pts_range_top)

        else:
            y_pts_range_top = np.linspace(y_pts_top[0],y_pts_top[-1],5)
            x_pts_range_top = torch.tensor(x_pts_range_top)
            y_pts_range_top = torch.tensor(y_pts_range_top)

        top_polygones = torch.cat((x_pts_range_top.unsqueeze(0),y_pts_range_top.unsqueeze(0)),0).T.reshape(-1)
        final_polygone = torch.cat((bottom_polygones,top_polygones),0)
        final_polygone.reshape(-1)

        pp = final_polygone
        interpolated_polygones.append(final_polygone)
    return interpolated_polygones

class MTHv2(Dataset):
    def __init__(self, mode, transform=transforms.ToTensor(), target_transform=None, path_to_data=None):
       # folder 
        if mode =='val':
            mode = 'train'
        self.mode = mode
        self.path_to_data =  os.path.join(path_to_data)
        with open(os.path.join(self.path_to_data, mode + ".txt")) as f:

            list_files = f.readlines()
            self.list_files = [ff.split('\n')[0] for ff in list_files]
        self.num_samples = len(self.list_files)
        self._transforms = transform

    def __len__(self):

        return self.num_samples 

    def __getitem__(self, idx):
        
        idx = idx
        path_img = self.list_files[idx]
        path_img = os.path.join(self.path_to_data, path_img.replace('MTHv2/',''))
        image = Image.open(path_img).convert("RGB")

        txt_file = path_img.replace("img","label_textline").replace("png","txt").replace("jpg","txt")
        size  = torch.tensor([image.size[0], image.size[1]], dtype=torch.int64)
        labels_txt =  open(txt_file).read().split("\n")[:-1]
        coordinates = [labels_txt[i].split(",")[1:] for i in range(len(labels_txt))]
        coordinates = np.array(coordinates).astype(int)
        sorted_coordinates = []
        for cc in coordinates:
            sorted_cc = list(sort_points_by_angle(cc.reshape(-1,2)).reshape(-1))
            sorted_coordinates.append(sorted_cc)
        
        interpolated_polygones = interpolate_polygones(sorted_coordinates)
        polygones = torch.stack(interpolated_polygones)
 
        labels = {}
        labels['size'] = size
        labels['orig_size'] = size
        labels['boxes'] = polygones.float()
        labels['labels'] = torch.zeros((polygones.shape[0],), dtype=torch.int64)
            
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


def build_MTHv2(image_set, args,path_to_data):
    transforms = make_coco_transforms(image_set, args=args)
    return MTHv2(image_set, transforms,path_to_data = path_to_data)

