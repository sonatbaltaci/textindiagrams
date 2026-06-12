if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))
import torch
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import datasets.transforms as T



current_dir = os.path.dirname(os.path.abspath(__file__))

if 'dataset' not in current_dir:
    current_dir = os.path.join(current_dir, 'dataset')
else:
    current_dir = os.path.join(current_dir, '.')

#with open(os.path.join(current_dir, 'config.json'), 'r') as f:
datasets_path = '/home/rbaena/datasets/'


class READ(Dataset):
    def __init__(self, mode, transform=transforms.ToTensor(), target_transform=None):
        """ 
        mode: train, valid, test
        """

        self.mode = mode
        self._transforms = transform
        ### load labels (text) from pickle file
        self.data = pickle.load(open(datasets_path + '/READ_segm/labels.pkl', "rb"))
        self.transform = transform
        self.target_transform = target_transform


    
    def __len__(self):
        return len(self.data[self.mode])


    def __getitem__(self, idx):
        data_item = self.data[self.mode][idx]
        #img = Image.open(datasets_path + '/cbdar2019/' + self.mode + '/' +data_item["idx"] + '.jpg')

        image = Image.open(os.path.join(datasets_path, 'READ_segm', self.mode, data_item["idx"]))
        labels = {}
        labels["labels"] = torch.tensor(data_item["labels"])
        labels["orig_size"]  = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
        labels["size"] = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
        labels["img_idx"] = torch.tensor([idx], dtype=torch.int64)
        labels["idx"] = torch.tensor([idx], dtype=torch.int64)

        labels["boxes"] =   torch.tensor(data_item["bboxes"], dtype=torch.float32)

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

def build_READ(image_set, args):
    transforms = make_coco_transforms(image_set, args=args)
    return READ(image_set, transforms)
