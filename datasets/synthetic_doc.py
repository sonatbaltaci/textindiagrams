
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))
import torch
import os
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import datasets.transforms as T
import math, random
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F


def random_deformation(image, polygons, grid_size=3, disp_std=0.2):
    """
    Applique une déformation non rigide à l'image et calcule la transformation
    correspondante sur les polygones en utilisant le champ de déplacement.

    Paramètres :
      - image : Tensor de shape (C, H, W)
      - polygons : Tensor de shape (N, 20) où chaque polygone est défini par [x1, y1, ..., x10, y10]
      - grid_size : nombre de points de contrôle par dimension (par exemple 3 pour une déformation globale)
      - disp_std : écart-type du déplacement aléatoire appliqué aux points de contrôle

    Retourne :
      - image_deformee : image déformée, Tensor de shape (C, H, W)
      - new_polygons : coordonnées transformées des polygones, Tensor de shape (N, 20)
    """
    C, H, W = image.shape
    nb_pt = len(polygons[0])//2
    # --- 1. Construction de la grille de contrôle ---
    # Grille de contrôle de base en coordonnées normalisées [-1, 1] (de taille grid_size x grid_size)
    y_lin = torch.linspace(-1, 1, grid_size, device=image.device)
    x_lin = torch.linspace(-1, 1, grid_size, device=image.device)
    grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing='ij')
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # shape: (grid_size, grid_size, 2)

    # --- 2. Ajout d'un déplacement aléatoire aux points de contrôle ---
    random_disp = torch.randn_like(base_grid) * disp_std
    control_grid = base_grid + random_disp  # grille de contrôle déformée

    # --- 3. Interpolation de la grille de contrôle vers une grille dense ---
    # On ajoute d'abord les dimensions batch et channel pour F.interpolate.
    control_grid = control_grid.unsqueeze(0).permute(0, 3, 1, 2)  # shape: (1, 2, grid_size, grid_size)
    dense_grid = F.interpolate(control_grid, size=(H, W), mode='bicubic', align_corners=True)  # (1, 2, H, W)
    dense_grid = dense_grid.permute(0, 2, 3, 1)  # (1, H, W, 2)

    # --- 4. Déformation de l'image ---
    # La grille dense est utilisée pour extraire l'image déformée.
    image_deformee = F.grid_sample(image.unsqueeze(0), dense_grid, mode='bilinear',
                                   padding_mode='border', align_corners=True)
    image_deformee = image_deformee.squeeze(0)  # (C, H, W)

    # --- 5. Calcul du champ de déplacement et transformation des polygones ---
    # a) Construction d'une grille d'identité pleine résolution en coordonnées normalisées.
    y_full = torch.linspace(-1, 1, H, device=image.device)
    x_full = torch.linspace(-1, 1, W, device=image.device)
    grid_y_full, grid_x_full = torch.meshgrid(y_full, x_full, indexing='ij')
    identity_grid = torch.stack([grid_x_full, grid_y_full], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # b) Calcul du champ de déplacement en coordonnées normalisées.
    # Pour chaque pixel, displacement = dense_grid - identity_grid
    dense_disp = dense_grid - identity_grid  # (1, H, W, 2)

    # c) Pour chaque point de polygone, on veut connaître le déplacement à son emplacement.
    # Convertir d'abord les coordonnées de polygones (en pixels) en coordonnées normalisées.
    polygons_pts = polygons.view(-1, nb_pt, 2)  # (N, 10, 2)
    norm_polygons = polygons_pts.clone()
    norm_polygons[..., 0] = (norm_polygons[..., 0] / (W )) * 2 - 1
    norm_polygons[..., 1] = (norm_polygons[..., 1] / (H)) * 2 - 1

    # d) Échantillonnage du déplacement au niveau des points du polygone.
    # Préparer les points pour grid_sample : (1, total_points, 1, 2)
    total_points = norm_polygons.numel() // 2
    sample_grid = norm_polygons.view(1, total_points, 1, 2)
    # Pour grid_sample, on met dense_disp sous forme (1, 2, H, W)
    dense_disp_t = dense_disp.permute(0, 3, 1, 2)
    disp_at_points = F.grid_sample(dense_disp_t, sample_grid, mode='bilinear',
                                   padding_mode='border', align_corners=True)
    # disp_at_points : (1, 2, total_points, 1) --> squeeze et permutation pour (total_points, 2)
    disp_at_points = disp_at_points.squeeze(0).squeeze(-1).permute(1, 0)

    # e) Calcul de la nouvelle position des points.
    # Attention : la grille dense utilisée pour déformer l'image réalise la transformation suivante :
    #   T (coordonnée cible) -> S = T + delta(T) (coordonnée source utilisée)
    # Pour inverser de manière approchée, on part du principe que T ≈ S - delta(S)
    # On calcule ainsi la nouvelle position T (en coordonnées normalisées) en soustrayant le déplacement.
    new_norm_points = norm_polygons.view(total_points, 2) - disp_at_points

    # f) Conversion des nouvelles coordonnées normalisées en pixels.
    new_norm_points_pixel = new_norm_points.clone()
    new_norm_points_pixel[:, 0] = ((new_norm_points_pixel[:, 0] + 1) / 2) * (W )
    new_norm_points_pixel[:, 1] = ((new_norm_points_pixel[:, 1] + 1) / 2) * (H )
    new_polygons = new_norm_points_pixel.view(polygons_pts.shape[0], nb_pt * 2)

    return image_deformee, new_polygons
def interpolate_polygone(polygones,num_points):
    interpolated_polygones =[]
    for pp in polygones:
        
        pp = torch.tensor(pp)
        bottom_polygones = pp[4:]
        x_pts_bottom = bottom_polygones[::2]
        y_pts_bottom = bottom_polygones[1::2]
        x_pts_range_bottom = np.linspace(x_pts_bottom[0],x_pts_bottom[-1],num_points//2)
        x_pts_bottom_sorted, idx_xpts_bottom = x_pts_bottom.sort()
        y_pts_bottom_sorted = y_pts_bottom[idx_xpts_bottom]
        if x_pts_range_bottom[0] != x_pts_range_bottom[-1]:
            y_pts_range_bottom = np.interp(x_pts_range_bottom,x_pts_bottom_sorted,y_pts_bottom_sorted)
        else:
            y_pts_range_bottom = np.linspace(y_pts_bottom[0],y_pts_bottom[-1],num_points//2)
        x_pts_range_bottom,y_pts_range_bottom = torch.tensor(x_pts_range_bottom),torch.tensor(y_pts_range_bottom)
        x_pts_range_bottom = torch.flip(x_pts_range_bottom, [0])
        y_pts_range_bottom = torch.flip(y_pts_range_bottom, [0])
        bottom_polygones = torch.cat((x_pts_range_bottom.unsqueeze(0),y_pts_range_bottom.unsqueeze(0)),0).T.reshape(-1)


        top_polygones = pp[:4]
        x_pts_top = top_polygones[::2]
        y_pts_top = top_polygones[1::2]
        x_pts_range_top = np.linspace(x_pts_top[0],x_pts_top[-1],num_points//2)
        if x_pts_range_top[0] != x_pts_range_top[-1]:
            x_pts_top_sorted, idx_xpts_top = x_pts_top.sort()
            y_pts_top_sorted = y_pts_top[idx_xpts_top]
            y_pts_range_top = np.interp(x_pts_range_top,x_pts_top_sorted,y_pts_top_sorted)
            x_pts_range_top,y_pts_range_top = torch.tensor(x_pts_range_top),torch.tensor(y_pts_range_top)

        else:
            y_pts_range_top = np.linspace(y_pts_top[0],y_pts_top[-1],num_points//2)
            x_pts_range_top = torch.tensor(x_pts_range_top)
            y_pts_range_top = torch.tensor(y_pts_range_top)
        x_pts_range_top = torch.flip(x_pts_range_top, [0])
        y_pts_range_top = torch.flip(y_pts_range_top, [0])
        top_polygones = torch.cat((x_pts_range_top.unsqueeze(0),y_pts_range_top.unsqueeze(0)),0).T.reshape(-1)
        final_polygone = torch.cat((bottom_polygones,top_polygones),0)
        final_polygone.reshape(-1)

        pp = final_polygone

        interpolated_polygones.append(final_polygone)
    #interpolations = self.linear_interpolation(bottom_line,6) # 6 points ie 12 coordinates
    try:
        interpolated_polygones = torch.stack(interpolated_polygones)
    except:
        # print(os.path.join(self.path_to_data, self.mode, str(idx) + '.jpg'))
        #raise ValueError('Error')
        raise ValueError('Error')
    return interpolated_polygones
from synthetic.document import SyntheticDocument
class Doc_synth(Dataset):
    def __init__(self, mode, transform=transforms.ToTensor(), target_transform=None, path_to_data=None,num_points = 10):
       # folder 
        self.mode = mode
        print(path_to_data)
        self.path_to_data =  path_to_data #os.path.join(path_to_data, mode)
        self._transforms = transform
        # list_files =  os.listdir(self.path_to_data)
        # #contains only .json files eg: 00_seg.json
        # self.img_idx = [f for f in list_files if f.endswith('.json')]
        # ## get the real idx eg 00_seg.json -> 00
        # self.img_idx = [f.split('_')[0] for f in self.img_idx]
        # self.img_idx.sort()
        if self.mode == 'train':
            self.num_samples = 2500
        else:
            self.num_samples = 500
        self.prop = 10
        self.num_points = num_points
        #check it's even
        assert self.num_points % 2 == 0


    def linear_interpolation(self,polygones, num_points):
        # Extract coordinates
        x1, y1, x2, y2 = polygones[:, 0], polygones[:, 1], polygones[:, 2], polygones[:, 3]
        
        # Calculate the slope
        slope = (y2 - y1) / (x2 - x1)
        
        # Calculate the x distances between the points
        dx = (x2 - x1) / (num_points + 1)
        
        # Generate x values for interpolation
        xi = x1.unsqueeze(1) + torch.arange(1, num_points + 1).float() * dx.unsqueeze(1)
        
        # Perform interpolation
        yi = y1.unsqueeze(1) + slope.unsqueeze(1) * (xi - x1.unsqueeze(1))
        
        # Stack x and y coordinates into a single tensor
        interpolated_points = torch.stack((xi, yi), dim=2)

        
        return interpolated_points
  
# Example usage:
    def __len__(self):

        return self.num_samples * self.prop

    def __getitem__(self, idx):
        idx = idx // self.prop
        image = Image.open(os.path.join(self.path_to_data, self.mode, str(idx) + '.jpg'))
        size  = torch.tensor([image.size[0], image.size[1]], dtype=torch.int64)

        labels_json= json.load(open(os.path.join(self.path_to_data, self.mode,str(idx) + '_seg.json')))
        # if len(labels_json['bboxes']) == 0:
        #     idx = np.random.randint(0, len(self.img_idx))
        # else:
        #     break

        labels = {}
        labels['size'] = size
        labels['orig_size'] = size
        # polygones = torch.tensor(labels_json['bboxes']) # (x1,y1,x2,y2, x3,y3,x4,y4)
        # bottom_line = polygones[:,:4] #x1,y1,x2,y2
        # top_line = polygones[:,4:] #x3,y3,x4,y4
        # ## interpolate two points

        # interpolations_bottom = self.linear_interpolation(bottom_line,self.num_points//2)
       
        # interpolations_bottom = interpolations_bottom.view(-1,self.num_points//2*2)

        # interpolations_top = self.linear_interpolation(top_line,self.num_points//2)
        # interpolations_top = interpolations_top.view(-1,self.num_points//2*2)
        # new_boxes = torch.cat((interpolations_bottom, interpolations_top),1)

        # # new_bottom_line = torch.zeros((polygones.shape[0],4+2*6))

        # new_bottom_line[:,:2] = polygones[:,:2]
  
        # new_bottom_line[:,2:2*6+2] = interpolations
        # new_bottom_line[:,2*6+2:] = polygones[:,2:4]

        # new_boxes = new_boxes#torch.cat((new_bottom_line, polygones[:,4:]),1)

        # labels['boxes'] = torch.tensor(new_boxes.clone().detach(), dtype=torch.float32)
        interpolated_polygones = interpolate_polygone(labels_json['bboxes'], self.num_points).float()
        image_tensor = ToTensor()(image)
        image_tensor, interpolated_polygones = random_deformation(image_tensor, interpolated_polygones, grid_size=np.random.randint(3,4),disp_std =random.uniform(0.0001, 0.02))
        # print(interpolated_polygones.tolist())
        # interpolated_polygones = interpolate_polygone(interpolated_polygones.tolist(), self.num_points)
        to_pil = ToPILImage()
        image =  to_pil(image_tensor)


     
        labels['boxes'] = interpolated_polygones.float()
        labels['labels'] = torch.tensor(labels_json['labels'], dtype=torch.int64)

    
        image, labels = self._transforms(image, labels)

        return image, labels

    def generate_synthetic_document(self, i, save_folder):
        print(i)
        while True:
            try:
                i  =str(i)
                kwargs = {'baseline_as_label': False, 'merged_labels': True, 'text_border_label': True}
                d = SyntheticDocument(**kwargs)
                img = d.to_image()
                label, labels_with_bboxes = d.to_label_as_array()
                img.save(os.path.join(save_folder, i + '.jpg'))
                if len(labels_with_bboxes["bboxes"]) == 0:
                    raise ValueError('No labels')
                annotation_dict = {'bboxes': labels_with_bboxes["bboxes"], 'labels': labels_with_bboxes["labels"]}
                with open(os.path.join(save_folder, i + '_seg.json'), 'w') as f:
                    json.dump(annotation_dict, f)
                break
            except Exception as e:
                print(e)
            continue
    
                
    def generate_synthetic_data(self):
        save_folder = os.path.join(self.path_to_data, self.mode)
        if not os.path.exists(self.path_to_data):
            os.makedirs(self.path_to_data)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # for k in range(self.num_samples):
        #     self.generate_synthetic_document(k,save_folder)
        import multiprocessing
        pool = multiprocessing.Pool() 
        results = [pool.apply_async(self.generate_synthetic_document, args=(k,save_folder)) for k in range(self.num_samples)]
        output = [p.get() for p in results]
        pool.close()
        print('DONE')


        


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


        if fix_size:
            return T.Compose([
                # T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                #augment_img,
                normalize,
            ])

        import datasets.sltransform as SLT
            
        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                
                ])
            ),

            
            #augment_img,
            normalize,
            T.blur,
        ])


def build_doc_synth(image_set, args,path_to_data):
    transforms = make_coco_transforms(image_set, args=args)
    return Doc_synth(image_set, transforms,path_to_data = path_to_data)

