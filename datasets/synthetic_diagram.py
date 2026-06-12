import os
import sys
import json
import signal
import torch
import multiprocessing
from PIL import Image
from time import sleep
from torch.utils.data import Dataset

import datasets.transforms as T
from datasets.EIDALatin import interpolate_polygons
from synthetic.synthetic_module.synthetic import SyntheticDiagram

class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def make_coco_transforms(image_set, args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = getattr(args, 'data_aug_scales', [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800])
    max_size = getattr(args, 'data_aug_max_size', 1333)
    
    overlap = getattr(args, 'data_aug_scale_overlap', None)
    if overlap is not None and overlap > 0:
        scales = [int(i * overlap) for i in scales]
        max_size = int(max_size * overlap)

    if image_set == 'train':
        return T.Compose([
            T.RemoveBoxes(),
            T.RandomApply(transforms=T.InstanceAwareCrop(), p=0.5),
            T.RandomResize(scales, max_size=max_size),
            normalize,
            T.blur,
        ])
    else:
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

class BaseSynthDataset(Dataset):
    def __init__(self, mode, transform, path_to_data, num_points=64):
        self.mode = mode
        self.path_to_data = path_to_data
        self._transforms = transform
        self.num_points = num_points
        self.prop = 10
        self.num_samples = 2500 if mode == 'train' else 10

    def __len__(self):
        return self.num_samples * self.prop

    def _get_raw_item(self, idx):
        """Internal helper to load files and handle the index logic."""
        actual_idx = idx // self.prop
        img_path = os.path.join(self.path_to_data, self.mode, f"{actual_idx}.jpg")
        json_path = os.path.join(self.path_to_data, self.mode, f"{actual_idx}_seg.json")
        
        image = Image.open(img_path)
        with open(json_path, 'r') as f:
            labels_json = json.load(f)
        
        size = torch.tensor([image.size[0], image.size[1]], dtype=torch.int64)
        return image, labels_json, size

    def generate_synthetic_data(self):
        save_folder = os.path.join(self.path_to_data, self.mode)
        os.makedirs(save_folder, exist_ok=True)

        pool = multiprocessing.Pool() 
        results = [pool.apply_async(self.generate_synthetic_diagram, args=(k, save_folder)) for k in range(self.num_samples)]
        [p.get() for p in results]
        pool.close()
        print('DONE')

    def generate_synthetic_diagram(self, i, save_folder, max_retries=10, timeout_seconds=10):
        attempt = 0
        is_latin = isinstance(self, Synthetic_Latin_Diagram)

        while attempt < max_retries:
            attempt += 1
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)

            try:
                diagram = SyntheticDiagram(is_latin=is_latin)
                image = diagram.to_image()

                annotations = diagram.get_annotation('test')

                polygones = annotations['polygones']
                if len(polygones) == 0:
                    raise ValueError("Empty polygons")

                data_to_save = {
                    'polygones': [p.flatten().tolist() for p in polygones],
                    'x_min': annotations['x_min'],
                    'y_min': annotations['y_min'],
                    'x_max': annotations['x_max'],
                    'y_max': annotations['y_max'],
                    'words': annotations.get('words', []),
                }
                image.save(os.path.join(save_folder, f"{i}.jpg"))
                with open(os.path.join(save_folder, f"{i}_seg.json"), 'w') as f:
                    json.dump(data_to_save, f)
                print(f"Diagram {i} saved.")
                signal.alarm(0)
                return True

            except TimeoutException:
                signal.alarm(0)
                sleep(1)
                continue

            except Exception as e:
                signal.alarm(0)
                print(f"Error generating diagram {i} (attempt {attempt}/{max_retries}): {e}")
                sleep(1)
                continue

        raise RuntimeError(f"Failed to generate diagram {i} after {max_retries} attempts")

class Synthetic_Diagram(BaseSynthDataset):
    def __getitem__(self, idx):
        while True:
            try:
                image, labels_json, size = self._get_raw_item(idx)
                
                # Process polygons
                interpolations = interpolate_polygons(labels_json['polygones'], self.num_points // 4)
                interpolations = torch.stack(interpolations)

                labels = {
                    'size': size,
                    'orig_size': size,
                    'boxes': interpolations.float(),
                    'labels': torch.ones(interpolations.shape[0], dtype=torch.int64),
                    'x_min': labels_json['x_min'],
                    'y_min': labels_json['y_min'],
                    'x_max': labels_json['x_max'],
                    'y_max': labels_json['y_max']
                }

                if len(labels['boxes']) > 0:
                    image, labels = self._transforms(image, labels)
                    return image, labels
                idx += self.prop
            except Exception:
                idx += self.prop

class Synthetic_Latin_Diagram(BaseSynthDataset):
    def __init__(self, mode, transform, path_to_data, num_points=64, num_classes=19):
        super().__init__(mode, transform, path_to_data, num_points)
        self.num_classes = num_classes
        self.classes_name = ['word','long','a','b','c','d','e','f','g','h','k','m','n','o','p','q','x','L','others']
        
        self.label2id = {label: idx for idx, label in enumerate(self.classes_name)}

    def get_characters_from_json(self, words):
        list_chars_id = []
        for label in words:
            if len(label) == 1:
                if label in self.classes_name:
                    class_id = self.label2id.get(label)
                elif label in ['O', '0']:
                    class_id = self.label2id.get('o')
                else:
                    class_id = self.label2id.get('others')
            else:
                nb_words = len(label.split(' '))
                if nb_words == 1:
                    class_id = self.label2id['word']
                elif nb_words > 1:
                    class_id = self.label2id['long']
                else:
                    class_id = self.label2id.get('others')
            list_chars_id.append(class_id)
        return torch.tensor(list_chars_id, dtype=torch.int64)

    def __getitem__(self, idx):
        while True:
            try:
                image, labels_json, size = self._get_raw_item(idx)
                
                interpolations = interpolate_polygons(labels_json['polygones'], self.num_points // 4)
                interpolations = torch.stack(interpolations)
                
                char_labels = self.get_characters_from_json(labels_json['words'])

                labels = {
                    'size': size,
                    'orig_size': size,
                    'boxes': interpolations.float(),
                    'labels': char_labels,
                    'x_min': labels_json['x_min'],
                    'y_min': labels_json['y_min'],
                    'x_max': labels_json['x_max'],
                    'y_max': labels_json['y_max']
                }
 
                if len(labels['boxes']) > 0:
                    image, labels = self._transforms(image, labels)
                    return image, labels
                idx += self.prop
            except Exception:
                idx += self.prop


def build_diagram_synth(image_set, args, path_to_data, is_latin):
    # Determine if we are using the Latin version based on args
    transforms = make_coco_transforms(image_set, args)
    
    if is_latin:
        return Synthetic_Latin_Diagram(
            image_set, transforms, path_to_data, 
            num_points=args.query_dim, num_classes=args.num_classes
        )
    else:
        return Synthetic_Diagram(
            image_set, transforms, path_to_data, 
            num_points=args.query_dim
        )