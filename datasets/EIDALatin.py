
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
import json
import base64
from PIL import Image
from io import BytesIO

from scipy.optimize import linear_sum_assignment

def get_polygons_from_json(data, n_vertices):
    list_polygons = []

    for shape in data['shapes']:
        a = np.array(shape['points'])
        if a.shape[0] % 2 != 0:
            # add one more point to make it divisible by 4
            a = np.vstack([a[0], a])
        list_polygons.append(a)
    interpolated_polygons = interpolate_polygons(list_polygons, n_vertices)   
    return interpolated_polygons

def segments_intersect(p1, p2, p3, p4):
    """Check if line segments (p1,p2) and (p3,p4) intersect."""
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def interpolate_polygons(list_polygons, n_vertices):
    
    if n_vertices == 2:
        bounding_boxes = []
        for poly in list_polygons:
            poly = torch.tensor(poly, dtype=torch.float32).flatten()
            nb_pts = poly.shape[0]
            assert nb_pts % 4 == 0, "Polygon should have equal number of top and bottom points"

            half = nb_pts // 2
            x_bottom, y_bottom = poly[:half][::2], poly[:half][1::2]
            x_top, y_top = poly[half:][::2], poly[half:][1::2]

            # Polygon vertices
            v0 = torch.stack([poly[0], poly[1]])
            v1 = torch.stack([poly[2], poly[3]])
            v2 = torch.stack([poly[half], poly[half + 1]])
            v3 = torch.stack([poly[half + 2], poly[half + 3]])

            x_min, x_max = torch.min(torch.cat([x_bottom, x_top])), torch.max(torch.cat([x_bottom, x_top]))
            y_min, y_max = torch.min(torch.cat([y_bottom, y_top])), torch.max(torch.cat([y_bottom, y_top]))

            # Bounding box corners
            box_corners = torch.tensor([
                [x_min, y_min],  # bottom-left
                [x_max, y_min],  # bottom-right
                [x_max, y_max],  # top-right
                [x_min, y_max],  # top-left
            ])

            poly_vertices = torch.stack([v0, v1, v2, v3])
            dists = torch.cdist(poly_vertices.unsqueeze(0), box_corners.unsqueeze(0)).squeeze(0)  # (4, 4)

            # Hungarian optimal assignment
            row_ind, col_ind = linear_sum_assignment(dists.numpy())
            ordered_corners = box_corners[col_ind]

            # Check for crossing edges (v0-v1 with v2-v3)
            if segments_intersect(ordered_corners[1], ordered_corners[2],
                                ordered_corners[0], ordered_corners[3]):
                # Swap last two to fix order
                ordered_corners[[2, 3]] = ordered_corners[[3, 2]]


            ordered_corners = ordered_corners.flatten()
            bounding_boxes.append(ordered_corners)
        return bounding_boxes

    else:
        interpolated_polygons = []

        for poly in list_polygons:
            poly = torch.tensor(poly, dtype=torch.float32).flatten()
            nb_pts = poly.shape[0]

            assert nb_pts % 4 == 0, "Polygon should have equal number of top and bottom points"

            half = nb_pts // 2
            x_bottom, y_bottom = poly[:half][::2], poly[:half][1::2]
            x_top, y_top = poly[half:][::2], poly[half:][1::2]        

            def resample_line(x, y, n):
                # Compute arc length along the polyline
                dx = np.diff(x)
                dy = np.diff(y)
                dist = np.sqrt(dx**2 + dy**2)
                arc = np.concatenate(([0], np.cumsum(dist)))

                # Interpolate based on arc length
                new_arc = np.linspace(0, arc[-1], n)
                new_x = np.interp(new_arc, arc, x)
                new_y = np.interp(new_arc, arc, y)
                return torch.tensor(new_x), torch.tensor(new_y)

            xb_new, yb_new = resample_line(x_bottom.numpy(), y_bottom.numpy(), n_vertices)
            xt_new, yt_new = resample_line(x_top.numpy(), y_top.numpy(), n_vertices)

            bottom_interp = torch.stack([xb_new, yb_new], dim=1).flatten()
            top_interp = torch.stack([xt_new, yt_new], dim=1).flatten()
            final_polygon = torch.cat([bottom_interp, top_interp], dim=0)

            interpolated_polygons.append(final_polygon)

        return interpolated_polygons

class EIDALatin(Dataset):
    def __init__(self, mode, transform=transforms.ToTensor(), path_to_data=None, num_classes=0):
       # folder 
        self.mode = mode
        self.path_to_data =  os.path.join(path_to_data)
        path_split = os.path.join(self.path_to_data, self.mode + ".txt")
        with open(path_split) as f:
            self.list_files = f.readlines()
        self.list_files = [x.strip() for x in self.list_files]
        
        self._transforms = transform
        txt_wrong = os.path.join(self.path_to_data, 'wrong_files.txt')
        list_wrong_files = []
        if os.path.exists(txt_wrong):
            with open(txt_wrong) as f:
                list_wrong_files = f.readlines()
            list_wrong_files = [x.strip() for x in list_wrong_files]
        # remove wrong files from list_files
        self.list_files = [x for x in self.list_files if x not in list_wrong_files]
    
        self.num_samples = len(self.list_files)
        self.classes = num_classes
        self.classes_check_name = ['word','long','a','b','c','d','e','f','g','h','k','m','n','o','p','q','x','L','others','symbol']
        self.label2id = {label: idx for idx, label in enumerate(self.classes_check_name)}

    def __len__(self):
        return self.num_samples 


    def get_characters_from_json(self, data):
        list_chars_id = []
        list_mask = []
        valid_labels = set(self.classes_check_name)
        for shape in data.get('shapes', []):
            label = shape.get('label', '').strip()
            if label in valid_labels:
                class_id = self.label2id[label]
                list_chars_id.append(class_id)
                list_mask.append(1)
                # class '?' is masked
            else:
                # Letters with low occurrence all belong to the “others” class
                class_id = self.label2id['others']
                list_chars_id.append(class_id)
                if label == '?':
                    list_mask.append(0)
                else:
                    list_mask.append(1)
        list_chars_id = torch.tensor(list_chars_id, dtype=torch.int64)
        list_mask = torch.tensor(list_mask, dtype=torch.int64)
        return list_mask, list_chars_id
    
    def __getitem__(self, idx):
        path_img = self.list_files[idx]
        path_img = os.path.join(self.path_to_data, 'data', path_img)

        json_file = path_img.replace(".jpg",".json")
    
        
        with open(json_file) as f:
            labels_json = json.load(f)
        image_data = base64.b64decode(labels_json['imageData'])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        size  = torch.tensor([image.size[0], image.size[1]], dtype=torch.int64)
        polygones = get_polygons_from_json(labels_json, n_vertices=16)
        list_mask,list_chars_id = self.get_characters_from_json(labels_json)
        if len(polygones) >1:
            polygones = torch.stack(polygones)
        else:
            polygones = torch.tensor(polygones[0]).unsqueeze(0)


        labels = {}
        labels['name'] = self.list_files[idx]
        labels['size'] = size
        labels['orig_size'] = size
        labels['boxes'] = polygones.float()
        #convert character to class id
        labels['labels'] = list_chars_id
        #if character is '?', it is masked = 0
        labels['mask_ce'] = list_mask 
        image, labels = self._transforms(image, labels)

        return image, labels

def make_coco_transforms(image_set, fix_size=False, args=None):
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
                    T.RandomResize([(max_size, max(scales))]),
                    normalize,
                ])

            return T.Compose([
                
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales, max_size=max_size),
                    
                    ])
                ),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                normalize,
                T.blur,
            ])
        if image_set in ['val', 'eval_debug', 'train_reg', 'test']:

            return T.Compose([
                T.RandomResize([max(scales)], max_size=max_size),
                normalize,

            ])

def build_diagram_latin(image_set, args,path_to_data):
    transforms = make_coco_transforms(image_set, args=args)
    return EIDALatin(image_set, transforms, path_to_data=path_to_data, num_classes=args.num_classes)

