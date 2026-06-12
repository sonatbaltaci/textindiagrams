# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.misc import interpolate
def blur(image,target):
    return F.gaussian_blur(image, kernel_size=3, sigma=(0.1, 3.0)), target

def crop(image, target, region):
    cropped_image = F.crop(image, *region)
    target = target.copy()
    i, j, h, w = region # i: top, j: left, h: height, w: width

    target["size"] = torch.tensor([h, w])

    fields = ["labels"]
    if "boxes" in target:
        boxes = target["boxes"]
        num_points = boxes.shape[1] // 2 
        boxes = boxes.view(-1, num_points, 2)
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        # zero-out top and left cropped boxes
        cropped_boxes = boxes - torch.as_tensor([j, i])
        # zero-out bottom and right cropped boxes
        cropped_boxes = torch.min(cropped_boxes, max_size) 
        cropped_boxes = cropped_boxes.clamp(min=0)

        target["boxes"] = cropped_boxes.view(-1, num_points * 2)
        fields.append("boxes")

    # keep boxes fully inside the image
    
    if "boxes" in target:
        boxes = target["boxes"]
        dim = boxes.shape[-1]//2
        boxes = boxes.view(-1, dim, 2)

        # --- determine which polygons are fully inside ---
        eps = 1e-6

        # Whether each vertex lies strictly inside the crop
        inside_x = (boxes[..., 0] > eps) & (boxes[..., 0] < w - eps)
        inside_y = (boxes[..., 1] > eps) & (boxes[..., 1] < h - eps)
        inside = inside_x & inside_y

        # Keep if at least one vertex is inside
        partially_or_fully_inside = inside.all(dim=1)

        # Drop polygons that lie fully on the border (all vertices on 0/w or 0/h)
        on_border_x = ((boxes[..., 0] <= eps) | (boxes[..., 0] >= w - eps))
        on_border_y = ((boxes[..., 1] <= eps) | (boxes[..., 1] >= h - eps))
        on_border = (on_border_x | on_border_y).all(dim=1)
        
        # if on_border.sum() > 0:
        #    raise ValueError("A polygon lies on the border after cropping.")
        
        # keep = partially_or_fully_inside & (~on_border)
        keep = partially_or_fully_inside
        target["boxes"] = boxes[keep].view(-1, dim * 2)

        for key in ["labels", "area", "iscrowd"]:
            if key in target:
                target[key] = target[key][keep]
    return cropped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.tensor([ratio_width, ratio_height]).repeat(boxes.shape[-1]//2)
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    w ,h= size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5


    # Remove boxes outside the image
    if "boxes" in target:
        boxes = target["boxes"]
        dim = boxes.shape[-1]//2
        boxes = boxes.view(-1, dim, 2)

        # --- determine which polygons are fully inside ---
        inside_x = (boxes[..., 0] >= 0) & (boxes[..., 0] <= h)
        inside_y = (boxes[..., 1] >= 0) & (boxes[..., 1] <= w)
        keep = (inside_x & inside_y).all(dim=1)  # polygon is valid only if *all* vertices inside

        target["boxes"] = boxes[keep].view(-1, dim*2)
        for key in ["labels", "area", "iscrowd"]:
            if key in target:
                target[key] = target[key][keep]
        

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class ResizeDebug(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)

class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

class RandomApply(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms(img, target)
        return img, target


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes /  torch.tensor([w, h], dtype=torch.float32).repeat(boxes.shape[-1]//2)
            target["boxes"] = boxes

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def instance_aware_crop(image, target):
    img_h, img_w = target["size"]

    polygons = target["boxes"]  # (N, 64)
    dim = polygons.shape[-1] // 2
    polys_reshaped = polygons.view(polygons.size(0), dim, 2)
    x_min = polys_reshaped[..., 0].min().item()
    y_min = polys_reshaped[..., 1].min().item()
    x_max = polys_reshaped[..., 0].max().item()
    y_max = polys_reshaped[..., 1].max().item()

    random_center_offset = random.randint(0, 30)
    top = int(max(y_min - random_center_offset, 0))
    left = int(max(x_min - random_center_offset, 0))

    crop_h = int(min(y_max - top + random.randint(0, 30), img_h - top))
    crop_w = int(min(x_max - left + random.randint(0, 30), img_w - left))

    # --- sanity check ---
    invalid = (
        crop_h <= 0
        or crop_w <= 0
        or top < 0
        or left < 0
        or top + crop_h > img_h
        or left + crop_w > img_w
    )
    if invalid:
        return image, target

    region = (top, left, crop_h, crop_w)
    return crop(image, target, region)


class InstanceAwareCrop(object):
    def __init__(self):
        pass

    def __call__(self, img, target):
        return instance_aware_crop(img, target)

def rotate(image, target, angle):
    """
    Rotates image and target by 90, 180, or 270 degrees counter-clockwise.
    """
    rotated_image = F.rotate(image, angle, expand=True)

    w, h = image.size
    
    target = target.copy()
    
    if angle == 90 or angle == 270:
        target["size"] = torch.tensor([w, h]) 
    else:
        target["size"] = torch.tensor([h, w]) 

    if "boxes" in target:
        boxes = target["boxes"]
        num_points = boxes.shape[1] // 2
        boxes = boxes.view(-1, num_points, 2)
        
        x = boxes[:, :, 0].clone()
        y = boxes[:, :, 1].clone()

        if angle == 90:
            # 90 CCW
            # Top-Right (w, 0) -> Top-Left (0, 0)
            # Formula: x' = y, y' = w - x
            new_x = y
            new_y = w - x
        elif angle == 180:
            # 180
            # Bottom-Right (w, h) -> Top-Left (0, 0)
            # Formula: x' = w - x, y' = h - y
            new_x = w - x
            new_y = h - y
        elif angle == 270:
            # 270 CCW (or -90)
            # Top-Left (0, 0) -> Bottom-Left (0, w)
            # Formula: x' = h - y, y' = x
            new_x = h - y
            new_y = x
        else:
            new_x = x
            new_y = y

        rotated_boxes = torch.stack((new_x, new_y), dim=2)
        target["boxes"] = rotated_boxes.view(-1, num_points * 2)

    if "masks" in target:
        target['masks'] = F.rotate(target['masks'], angle, expand=True)

    return rotated_image, target


class RandomRotation(object):
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img, target):
        angle = random.choice(self.angles)
        if angle == 0:
            return img, target
        return rotate(img, target, angle)
    

class RemoveBoxes(object):
    def __init__(self):
        pass
    def __call__(self, img, target):
        # Remove boxes generated outside the image
        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            dim = boxes.shape[-1]//2
            boxes = boxes.view(-1, dim, 2)

            # --- determine which polygons are fully inside ---
            inside_x = (boxes[..., 0] >= 0) & (boxes[..., 0] <= target["size"][0])
            inside_y = (boxes[..., 1] >= 0) & (boxes[..., 1] <= target["size"][1])
            keep = (inside_x & inside_y).all(dim=1)
            target["boxes"] = boxes[keep].view(-1, dim*2)
            for key in ["labels", "area", "iscrowd"]:
                if key in target:
                    target[key] = target[key][keep]
        return img, target