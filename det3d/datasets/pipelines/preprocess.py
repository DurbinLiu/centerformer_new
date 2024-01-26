import numpy as np

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES
from torch import torch

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B,), angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, :3], rot_matrix)
    points_rot = np.concatenate([points_rot, points[:, :, 3:]], axis=-1)
    return points_rot
def get_points_in_gt_boxes_mask(points, gt_boxes, return_point_gt_idx):
    # gt_boxes = gt_boxes[:1]

    N, M = points.shape[0], gt_boxes.shape[0]

    repeat_points = points.repeat(M, axis=0)  # [NXM, 4]
    repeat_gt_boxes = np.expand_dims(gt_boxes, axis=0).repeat(N, axis=0).reshape(-1, 7)  # [NxM, 7]

    xyz_local = repeat_points[:, :3] - repeat_gt_boxes[:, :3]  # [NxM, 7]
    xyz_local = rotate_points_along_z(xyz_local[:, None, :], -repeat_gt_boxes[:, 6]).squeeze(axis=1)
    xy_local = xyz_local[:, :2]
    lw = repeat_gt_boxes[:, 3:5]
    is_in_gt_matrix = ((xy_local <= lw / 2) & (xy_local >= -lw / 2)).all(axis=-1).reshape(N, M)  # (N, M)
    is_in_gt = np.max(is_in_gt_matrix, axis=-1)  # (N,

    if return_point_gt_idx:
        points_idx, gt_boxes_idx = np.where(is_in_gt_matrix == 1)
        return points_idx, gt_boxes_idx

    return is_in_gt, None

@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_noise = cfg.get('global_translate_noise', 0)
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None 
                
            self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        elif res["type"] in ["NuScenesDataset"]:
            points = res["lidar"]["combined"]
        else:
            raise NotImplementedError

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }

        if self.mode == "train" and not self.no_augmentation:
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )


                    points = np.concatenate([sampled_points, points], axis=0)

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
            
            gt_dict["gt_boxes"], points = prep.global_rotation(
                gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
            )
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise
            )
            gt_dict["gt_boxes"], points = prep.global_translate_v2(
                gt_dict["gt_boxes"], points, noise_translate=self.global_translate_noise
            )
        elif self.no_augmentation:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes


        if self.shuffle_points:
            np.random.shuffle(points)

        res["lidar"]["points"] = points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num, int) else cfg.max_voxel_num

        self.double_flip = cfg.get('double_flip', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            res["lidar"]["annotations"] = gt_dict
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]
        # max_voxels = self.max_voxel_num[1]

        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["lidar"]["points"], max_voxels=max_voxels 
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )

        double_flip = self.double_flip and (res["mode"] != 'train')
        print(double_flip, res['mode'])

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )            

        return res, info

def flatten(box):
    return np.concatenate(box, axis=0)

def merge_multi_group_label(gt_classes, num_classes_by_task): 
    num_task = len(gt_classes)
    flag = 0 

    for i in range(num_task):
        gt_classes[i] += flag 
        flag += num_classes_by_task[i]

    return flatten(gt_classes)


@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.corner_prediction = assigner_cfg.get('corner_prediction', False)
        self.gt_kernel_size = assigner_cfg.get('gt_kernel_size', 1)
        print('use gt label assigning kernel size ', self.gt_kernel_size)
        self.cfg = assigner_cfg
        self.with_auxtask = assigner_cfg.get('with_AuxTask', False)
        self.use_ignore = False # 其实相关的都没用到，一直是False
        self.grid_size = [(assigner_cfg['pc_range'][i+3] - assigner_cfg['pc_range'][i]) / assigner_cfg['voxel_size'][i] for i in range(3)]

    def __call__(self, res, info):
        max_objs = self._max_objs
        gt_kernel_size = self.gt_kernel_size
        window_size = gt_kernel_size**2
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]

        example = {}

        if res["mode"] == "train":
            if 'pc_range' in self.cfg:
                pc_range = np.array(self.cfg['pc_range'], dtype=np.float32)
                voxel_size = np.array(self.cfg['voxel_size'], dtype=np.float32)
                grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
                grid_size = np.round(grid_size).astype(np.int64)
            elif 'voxels' in res['lidar']:
                # Calculate output featuremap size
                grid_size = res["lidar"]["voxels"]["shape"] 
                pc_range = res["lidar"]["voxels"]["range"]
                voxel_size = res["lidar"]["voxels"]["size"]
            else:
                raise NotImplementedError("range and size configuration are missing in the config!")
            # BEV map down sample scale
            ds_factor=self.out_size_factor
            # get width and height
            W,H=(pc_range[3] - pc_range[0]) / voxel_size[0]/ ds_factor, (pc_range[4] - pc_range[1]) / voxel_size[1]/ ds_factor
            W,H=np.round(W).astype(int),np.round(H).astype(int)
            feature_map_size = grid_size[:2] // self.out_size_factor
            
            gt_dict = res["lidar"]["annotations"]

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats = [], [], [], [], []
            if self.corner_prediction:
                corners = []

            for idx, task in enumerate(self.tasks):
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                              dtype=np.float32)
                
                if self.corner_prediction:
                    corner = np.zeros((1, feature_map_size[1], feature_map_size[0]), dtype=np.float32)

                if res['type'] == 'NuScenesDataset':
                    # [reg, hei, dim, vx, vy, rots, rotc]
                    anno_box = np.zeros((max_objs*window_size, 10), dtype=np.float32)
                elif res['type'] in ['WaymoDataset','WaymoDataset_multi_frame']:
                    anno_box = np.zeros((max_objs*window_size, 10), dtype=np.float32) 
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")

                ind = np.zeros((max_objs*window_size), dtype=np.int64)
                mask = np.zeros((max_objs*window_size), dtype=np.uint8)
                cat = np.zeros((max_objs*window_size), dtype=np.int64)

                num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)  

                for k in range(num_objs):
                    cls_id = gt_dict['gt_classes'][idx][k] - 1

                    w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                              gt_dict['gt_boxes'][idx][k][5]
                    w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))

                        # be really careful for the coordinate system of your box annotation. 
                        x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                  gt_dict['gt_boxes'][idx][k][2]

                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)  
                        ct_int = ct.astype(np.int32)

                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue 

                        draw_gaussian(hm[cls_id], ct, radius)
                        if self.corner_prediction:
                            radius = radius//2
                            # draw four corner and center
                            dim = np.array([w, l], dtype=np.float32)  
                            rot = np.array([gt_dict['gt_boxes'][idx][k][8]], dtype=np.float32)  
                            corner_keypoints = box_np_ops.center_to_corner_box2d(ct[np.newaxis,:],dim[np.newaxis,:],rot)
                            draw_gaussian(corner[0], ct, radius)
                            draw_gaussian(corner[0], (corner_keypoints[0, 0] + corner_keypoints[0, 1])/2, radius)
                            draw_gaussian(corner[0], (corner_keypoints[0, 2] + corner_keypoints[0, 3])/2, radius)
                            draw_gaussian(corner[0], (corner_keypoints[0, 0] + corner_keypoints[0, 3])/2, radius)
                            draw_gaussian(corner[0], (corner_keypoints[0, 1] + corner_keypoints[0, 2])/2, radius)

                        new_idx = k
                        x, y = np.arange(ct_int[0]-gt_kernel_size//2,ct_int[0]+1+gt_kernel_size//2), np.arange(ct_int[1]-gt_kernel_size//2,ct_int[1]+1+gt_kernel_size//2)
                        x, y = np.meshgrid(x, y)
                        x = x.reshape(-1)
                        y = y.reshape(-1)

                        for j in range(window_size):

                            cat[new_idx*window_size+j] = cls_id
                            ind[new_idx*window_size+j] = y[j] * feature_map_size[0] + x[j]
                            mask[new_idx*window_size+j] = 1

                            if res['type'] == 'NuScenesDataset': 
                                vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                                rot = gt_dict['gt_boxes'][idx][k][8]
                                anno_box[new_idx*window_size+j] = np.concatenate(
                                    (ct - (x[j], y[j]), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                    np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                            elif res['type'] in ['WaymoDataset','WaymoDataset_multi_frame']:
                                vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                                rot = gt_dict['gt_boxes'][idx][k][-1]
                                anno_box[new_idx*window_size+j] = np.concatenate(
                                (ct - (x[j], y[j]), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                            else:
                                raise NotImplementedError("Only Support Waymo and nuScene for Now")

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)
                if self.corner_prediction:
                    corners.append(corner)

            # used for two stage code 
            boxes = flatten(gt_dict['gt_boxes'])
            classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)

            if res["type"] == "NuScenesDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            elif res['type'] in ['WaymoDataset','WaymoDataset_multi_frame']:
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            else:
                raise NotImplementedError()

            boxes_and_cls = np.concatenate((boxes, 
                classes.reshape(-1, 1).astype(np.float32)), axis=1)
            num_obj = len(boxes_and_cls)
            assert num_obj <= max_objs
            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
            boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
            gt_boxes_and_cls[:num_obj] = boxes_and_cls

            example.update({'gt_boxes_and_cls': gt_boxes_and_cls})

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
            if self.corner_prediction:
                example.update({'corners': corners})
            if self.with_auxtask:
                aux_label = self.get_aux_targets(res, info)
                example['aux_label'] = aux_label[0]
                example['aux_reg'] = aux_label[1]
        else:
            pass

        res["lidar"]["targets"] = example

        return res, info

    def get_aux_targets(self, res, info):
            gt_bboxes_3d = res['lidar']['annotations']['gt_boxes'][0]     # N*C : 40*9  only applied to bs=1, 
            gt_boxes = gt_bboxes_3d[:, :7]  # cx cy cz l w h heading vx vy
            # gt_labels_3d = res['lidar']['annotations']['gt_classes'][0]   #  N  : 40
            cur_frame_pts_len = res['lidar']['points_num']
            points = res['lidar']['points'][:cur_frame_pts_len]
            # points = res['lidar']['points'][0][:cur_frame_pts_len]  # 当前帧点：第一个combine的前面部分,其实整个res['points']都是当前帧重复的
            # is_ignore = torch.empty((len(gt_bboxes_3d), )).fill_(-1)

            voxel_size = [ size*2 for size in self.cfg['voxel_size']]  # two times down sampling
            # mask_pts = points[:, -1] == 0
            # points = points[mask_pts]
            # ignore_mask = is_ignore == 1
            
            # gt_boxes = gt_bboxes_3d[~ignore_mask][:, :7]
            # gt_boxes_ignore = gt_bboxes_3d[ignore_mask][:, :7]

            length, width = np.array(self.grid_size[:2]) / 2
            length, width = int(length), int(width)
            if len(gt_boxes) == 0:
                aux_label1 = torch.zeros([1, width, length], dtype=torch.int32) - 1
                aux_label2 = torch.zeros([2, width, length], dtype=torch.float32) - 1
                return (aux_label1, aux_label2)
            voxel_label = -np.ones((length * width), dtype='int32')
            self.pc_range = np.array(self.cfg['pc_range'])
            # import pdb;pdb.set_trace()
            x, y = points[:, 0], points[:, 1]
            l = (x - self.pc_range[0]) // voxel_size[0]
            w = (y - self.pc_range[1]) // voxel_size[1]
            is_in_pc_range = (l < length) & (l >= 0) & (w < width) & (w >= 0)
            points = points[is_in_pc_range]   # 183680
            l, w = l[is_in_pc_range], w[is_in_pc_range]
            non_empty_idx = (l * width + w).astype('int32') # 183680
            voxel_label[non_empty_idx] = 0
            return_point_gt_idx = True
            is_in_gt, gt_boxes_idx = get_points_in_gt_boxes_mask(points, gt_boxes, return_point_gt_idx)  # 10821

            #set front ground points lable as 1
            # self.use_ignore = FAlse
            if self.use_ignore and len(gt_boxes_ignore) > 0:
                is_in_gt_ignored, _ = get_points_in_gt_boxes_mask(points, gt_boxes_ignore, return_point_gt_idx)
                l_ignored, w_ignored = l[is_in_gt_ignored], w[is_in_gt_ignored]
                fg_ignored_idx = (l_ignored * width + w_ignored).astype('int32')
                voxel_label[fg_ignored_idx] = 1

            l, w = l[is_in_gt], w[is_in_gt]   # 10821
            fg_idx = (l * width + w).astype('int32')
            voxel_label[fg_idx] = 1  # 565504 752*752

            voxel_label = voxel_label.reshape((-1, length, width))
            aux_label = (torch.from_numpy(voxel_label.transpose(0, 2, 1)),)  # (1,752,752)
            if return_point_gt_idx:
                # init dxdy offset
                offset_label = -np.ones((2, length * width), dtype='float32')

                # convert to real world coordinates
                voxel_center = np.stack([
                    l * voxel_size[0] + self.pc_range[0],
                    w * voxel_size[1] + self.pc_range[1]])
                gt_center = gt_boxes[gt_boxes_idx, :2].transpose((1, 0))  # gt box center x & y
                # offset label
                if len(fg_idx):
                    offset_label[:, fg_idx] = gt_center - voxel_center    # offset between fg box and voxel
                offset_label = offset_label.reshape((-1, length, width))

                aux_label += (torch.from_numpy(offset_label.transpose(0, 2, 1)),)
            # aux_label:((1,752,752) , (2,752, 752) )
            return aux_label