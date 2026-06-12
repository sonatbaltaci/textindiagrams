if __name__ == "__main__":
    import os, sys

    sys.path.append("../")
import torch
import os
import numpy as np
import json
from PIL import Image
from torchvision import transforms
import datasets.transforms as T
from torch.utils.data import Dataset

current_directory = os.getcwd()
print("curr", current_directory)
from scipy.optimize import linear_sum_assignment


# display files in current directory
def get_polygons_from_json(data, n_vertices):
    list_polygons = []

    for shape in data["shapes"]:
        a = np.array(shape["points"])
        if a.shape[0] % 2 != 0:
            # add one more point to make it divisible by 4
            a = np.vstack([a[0], a])
        list_polygons.append(a)
    interpolated_polygons = interpolate_polygons(list_polygons, n_vertices)
    return interpolated_polygons


def segments_intersect(p1, p2, p3, p4):
    """Check if line segments (p1,p2) and (p3,p4) intersect."""

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def interpolate_polygons(list_polygons, n_vertices):

    if n_vertices == 2:
        bounding_boxes = []
        for poly in list_polygons:
            poly = torch.tensor(poly, dtype=torch.float32).flatten()
            nb_pts = poly.shape[0]
            assert (
                nb_pts % 4 == 0
            ), "Polygon should have equal number of top and bottom points"

            half = nb_pts // 2
            x_bottom, y_bottom = poly[:half][::2], poly[:half][1::2]
            x_top, y_top = poly[half:][::2], poly[half:][1::2]

            # Polygon vertices
            v0 = torch.stack([poly[0], poly[1]])
            v1 = torch.stack([poly[2], poly[3]])
            v2 = torch.stack([poly[half], poly[half + 1]])
            v3 = torch.stack([poly[half + 2], poly[half + 3]])

            x_min, x_max = torch.min(torch.cat([x_bottom, x_top])), torch.max(
                torch.cat([x_bottom, x_top])
            )
            y_min, y_max = torch.min(torch.cat([y_bottom, y_top])), torch.max(
                torch.cat([y_bottom, y_top])
            )

            # Bounding box corners
            box_corners = torch.tensor(
                [
                    [x_min, y_min],  # bottom-left
                    [x_max, y_min],  # bottom-right
                    [x_max, y_max],  # top-right
                    [x_min, y_max],  # top-left
                ]
            )

            poly_vertices = torch.stack([v0, v1, v2, v3])
            dists = torch.cdist(
                poly_vertices.unsqueeze(0), box_corners.unsqueeze(0)
            ).squeeze(
                0
            )  # (4, 4)

            # Hungarian optimal assignment
            row_ind, col_ind = linear_sum_assignment(dists.numpy())
            ordered_corners = box_corners[col_ind]

            # Check for crossing edges (v0-v1 with v2-v3)
            if segments_intersect(
                ordered_corners[1],
                ordered_corners[2],
                ordered_corners[0],
                ordered_corners[3],
            ):
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

            assert (
                nb_pts % 4 == 0
            ), "Polygon should have equal number of top and bottom points"

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

            xb_new, yb_new = resample_line(
                x_bottom.numpy(), y_bottom.numpy(), n_vertices
            )
            xt_new, yt_new = resample_line(x_top.numpy(), y_top.numpy(), n_vertices)

            bottom_interp = torch.stack([xb_new, yb_new], dim=1).flatten()
            top_interp = torch.stack([xt_new, yt_new], dim=1).flatten()
            final_polygon = torch.cat([bottom_interp, top_interp], dim=0)

            interpolated_polygons.append(final_polygon)

        return interpolated_polygons


class EIDA(Dataset):
    def __init__(self, mode, transform=None, path_to_data=None, n_vertices=16):
        """
        Args:
            mode: 'train', 'val', or 'test'
            transform: transformation callable
            path_to_data: path to EIDA root
        """
        self.mode = mode
        self.path_to_data = os.path.join(path_to_data, mode)
        self._transforms = transform if transform else transforms.ToTensor()

        # List of all image files that have a matching .json file
        self.list_files = sorted(
            [
                f
                for f in os.listdir(self.path_to_data)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
                and os.path.exists(
                    os.path.join(self.path_to_data, f.rsplit(".", 1)[0] + ".json")
                )
            ]
        )

        self.num_samples = len(self.list_files)
        if self.num_samples == 0:
            raise RuntimeError(
                f"No valid image/JSON pairs found in {self.path_to_data}"
            )

        self.num = n_vertices * 4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_name = self.list_files[idx]

        img_path = os.path.join(self.path_to_data, img_name)
        json_path = os.path.join(
            self.path_to_data, img_name.rsplit(".", 1)[0] + ".json"
        )

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load LabelMe JSON
        with open(json_path, "r") as f:
            labelme_data = json.load(f)

        polygons = get_polygons_from_json(labelme_data, n_vertices=self.num // 4)

        if len(polygons) == 0:
            # If no polygons found, create empty tensors
            boxes = torch.zeros((0, 64), dtype=torch.float32)

        else:
            # Stack all polygons; pad to max length if needed for batching later
            boxes = torch.stack(polygons)

        size = torch.tensor([image.size[0], image.size[1]], dtype=torch.int64)

        labels = {
            "name": img_name,
            "size": size,
            "orig_size": size,
            "boxes": boxes.float(),
            "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),
        }

        while True:
            try:
                image, labels = self._transforms(image, labels)
                break
            except ValueError:
                continue
        assert labels["boxes"].shape[1] == self.num, labels["boxes"].shape
        return image, labels


def make_coco_transforms(image_set, args):
    fix_size = args.fix_size
    if image_set in ["val", "test"]:
        fix_size = True
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    # update args from config files
    scales = getattr(args, "data_aug_scales", scales)
    max_size = getattr(args, "data_aug_max_size", max_size)
    scales2_resize = getattr(args, "data_aug_scales2_resize", scales2_resize)
    scales2_crop = getattr(args, "data_aug_scales2_crop", scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, "data_aug_scale_overlap", None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i * data_aug_scale_overlap) for i in scales]
        max_size = int(max_size * data_aug_scale_overlap)
        scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]

    if fix_size:
        return T.Compose(
            [
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ]
        )

    return T.Compose(
        [
            T.RandomResize(scales, max_size=max_size),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose(
                    [
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ]
                ),
            ),
            normalize,
            T.blur,
        ]
    )


def build_diagram(image_set, args, path_to_data):
    transforms = make_coco_transforms(image_set, args=args)
    return EIDA(
        image_set, transforms, path_to_data=path_to_data, n_vertices=args.query_dim // 4
    )
