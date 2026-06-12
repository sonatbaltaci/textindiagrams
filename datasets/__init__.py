# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
import os

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

# Get current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

def build_dataset(image_set, args):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file =='document':
        from .synthetic_doc import build_doc_synth
        return build_doc_synth(image_set, args, curr_dir+'/../data/synthetic_doc')
    if args.dataset_file =='MTHv2':
        from .MTHv2 import build_MTHv2
        return build_MTHv2(image_set, args, curr_dir+'/../data/MTHv2')
    if args.dataset_file =='ICDAR2019CHINESE':
        from .icdar19chinese import build_ICDAR2019CHINESE
        return build_ICDAR2019CHINESE(image_set, args, curr_dir+'/../data/ICDAR2019CHINESE')
    if args.dataset_file == 'cbdar2019':
        from .cbdar2019 import build_cbdar2019
        return build_cbdar2019(image_set, args)
    if args.dataset_file =='synthetic_diagram':
        from .synthetic_diagram import build_diagram_synth
        return build_diagram_synth(image_set, args, curr_dir+'/../data/synthetic_diagram', is_latin=False)
    if args.dataset_file =='synthetic_latin_diagram':
        from .synthetic_diagram import build_diagram_synth
        return build_diagram_synth(image_set, args, curr_dir+'/../data/synthetic_latin_diagram', is_latin=True)
    if args.dataset_file =='eida':
        from .EIDA import build_diagram
        return build_diagram(image_set, args, curr_dir+'/../data/EIDA')
    if args.dataset_file =='latin_eida':
        from .EIDALatin import build_diagram_latin
        return build_diagram_latin(image_set, args, curr_dir+'/../data/EIDALatin/')

    
    raise ValueError(f'dataset {args.dataset_file} not supported')
