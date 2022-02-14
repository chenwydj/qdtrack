# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info

from pdb import set_trace as bp

import numpy as np
class Features:
    def __init__(self):
        self.outputs = []

    # def __call__(self, module, module_in):
        # self.outputs.append(module_in[0])
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[0])
        # self.outputs.append(module_out[0].unsqueeze(0))

    def clear(self):
        self.outputs = []


def mask_feature(module, input, output):
    _, _, h, w = output.shape
    mask = torch.nn.functional.interpolate(module.mask, size=(h, w))
    return output * mask


from torchvision import transforms
inv_normalizer = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
# BACKGROUND = -2.1179


THRESHOLD = 0.01 # detection confidence
RATIO = 0.2
GRID_H = GRID_W = 60
from shapely.geometry import box
def box_iou(loc, bboxes, complexity_type='intersection'):
    assert complexity_type in ["iou", "intersection"]
    rect_target = box(*loc)
    complexity = 0
    for rect in bboxes:
        # https://stackoverflow.com/questions/39049929/finding-the-area-of-intersection-of-multiple-overlapping-rectangles-in-python
        if complexity_type == "intersection":
            complexity += rect_target.intersection(rect).area
        else:
            # complexity += (rect_target.intersection(rect).area / rect_target.union(rect).area)
            complexity += (rect_target.intersection(rect).area / max(min(rect_target.area, rect.area), 1e-3))
    return complexity


# detection prediction from frame t-1 as complexity
def scan_complexity(image, bboxes, grid_h, grid_w, complexity_type="intersection"):
    areas = []
    locations = []
    complexities = []
    assert len(image.shape) == 4
    img_h, img_w = image.shape[2:]
    # grid_h, grid_w = int(np.ceil(img_h / GRID_H)), int(np.ceil(img_w / GRID_W))
    start_h = 0
    while start_h < img_h:
        end_h = min(img_h, start_h + grid_h)
        start_w = 0
        while start_w < img_w:
            end_w = min(img_w, start_w + grid_w)
            # complexities.append(box_iou([start_h, start_w, end_h-start_h, end_w-start_w], bboxes))
            complexities.append(box_iou([start_w, start_h, end_w, end_h], bboxes, complexity_type=complexity_type))
            # complexities.append(box_iou([start_h, start_w, end_h, end_w], bboxes))
            # complexities.append(box_iou([start_w, end_h, end_w, start_h], bboxes))
            locations.append([start_h, end_h, start_w, end_w])
            areas.append((end_h-start_h)*(end_w-start_w))
            start_w += grid_w
        start_h += grid_h
    locations_sorted = [x for _, x in sorted(zip(complexities, locations))]
    areas_sorted = [x for _, x in sorted(zip(complexities, areas))]
    return locations_sorted, areas_sorted, locations, areas, complexities


def bbox_zeros(image, bboxes, ratio, grid_h, grid_w, complexity_type="intersection"):
    img_h, img_w = image.shape[2:]
    locations_sorted, areas_sorted, locations, areas, complexities = scan_complexity(image, bboxes, grid_h, grid_w, complexity_type=complexity_type)
    assert sum(areas_sorted) == img_h*img_w, "sum(areas_sorted) = %d v.s. img_h*img_w = %d"%(sum(areas_sorted), img_h*img_w)
    ratios = np.cumsum(areas_sorted) / (img_h*img_w)
    threshold = (ratios < ratio).sum()
    for i in range(threshold):
        start_h, end_h, start_w, end_w = locations_sorted[i]
        # image[:, :, start_h:end_h, start_w:end_w] = BACKGROUND
        image[:, 0, start_h:end_h, start_w:end_w] = -0.485/0.229
        image[:, 1, start_h:end_h, start_w:end_w] = -0.456/0.224
        image[:, 2, start_h:end_h, start_w:end_w] = -0.406/0.225
    return image


def summarize_bbox(bbox_results):
    bboxes = []
    for l1 in range(len(bbox_results)):
        for l2 in range(len(bbox_results[l1])):
            if bbox_results[l1][l2][4] > THRESHOLD:
                # rect = box(*bbox_results[l1][l2][:4])
                # rect = box(bbox_results[l1][l2][0], bbox_results[l1][l2][1], bbox_results[l1][l2][2] - bbox_results[l1][l2][0], bbox_results[l1][l2][3] - bbox_results[l1][l2][1])
                # rect = box(bbox_results[l1][l2][1], bbox_results[l1][l2][0], bbox_results[l1][l2][3], bbox_results[l1][l2][2])
                rect = box(bbox_results[l1][l2][0], bbox_results[l1][l2][1], bbox_results[l1][l2][2], bbox_results[l1][l2][3])
                # rect = box(bbox_results[l1][l2][1], bbox_results[l1][l2][2], bbox_results[l1][l2][3], bbox_results[l1][l2][0])
                bboxes.append(rect)
    return bboxes


def drop_by_bbox(data, results, grid_h=GRID_H, grid_w=GRID_W, ratio=RATIO, complexity_type="intersection"):
    import time
    if len(results) > 0:
        start_time = time.time()
        bboxes = summarize_bbox(results["bbox_results"])
        # print("summarize_bbox", time.time() - start_time)
        # only support batch_size = 1 for now, since bbox is only from one frame
        # data["img"][0] is still BCHW, B = 1
        start_time = time.time()
        data["img"][0] = bbox_zeros(data["img"][0], bboxes, ratio, grid_h, grid_w, complexity_type=complexity_type)
        # print("bbox_zeros", time.time() - start_time)
    return data


def merge_complexities(complexities=None, complexities_pre=None, merge=True, norm=True, comp_type="mean"):
    '''
    complexities_pre: complexity computed by image content
    complexities: complexity computed by previous frame
    merge: flag to decide merge complexity lists
    norm: flag to decide norm the complexity list
    comp_type: how to do compose
    '''
    assert complexities is not None or complexities_pre is not None
    if not merge:
        return complexities_pre
    if complexities is None:
        return complexities_pre
    if complexities_pre is None:
        return complexities
    else:
        complexities_composed = list()
        complexities_pre = complexities_pre.tolist()
        assert len(complexities) == len(complexities_pre)

        # import matplotlib.pyplot as plt
        # plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
        # plt.hist(complexities, bins=50)
        # plt.savefig("distrib_prevframe_nonorm.png")
        # bp()
        if norm:
            complexities_pre = [comp/max(max(complexities_pre), 1e-3) for comp in complexities_pre]
            complexities = [comp/max(max(complexities), 1e-3) for comp in complexities]

        if comp_type == "mean":
            complexities_composed = [(abs(img) + abs(box))/2 for img, box in zip(complexities, complexities_pre)]
        elif comp_type == "multiply":
            complexities_composed = [(abs(img) * abs(box)) for img, box in zip(complexities, complexities_pre)]
        else:
            raise NotImplementedError

        return complexities_composed


def apply_dropping(data, results, locations_pre=None, areas_pre=None, complexity_pre=None, grid_h=GRID_H, grid_w=GRID_W, ratio=RATIO, complexity_type="intersection"):
    """
    data: dict of 'img' and 'img_metas' for current frame
    results: predictions from last frame
    locations_pre: patches cropped during preprocessing in dataloader
    areas_pre: areas of patches cropped during preprocessing in dataloader
    complexity_pre: complexities of patches from preprocessing in dataloader (by measuring nature image complexity)
    """
    img = data["img"].data[0]
    mask = torch.ones(img.shape[2], img.shape[3]).to(img.device)
    # aa = data["img"].data[0].squeeze(0).permute(1,2,0).cpu().numpy()
    # aa = aa - aa.min()
    # bb = (aa / aa.max() * 255).astype(np.uint8)
    # cv2.imwrite("img_ori.png", cv2.cvtColor(bb,cv2.COLOR_RGB2BGR))

    img_h, img_w = img.shape[2:]
    complexities = None
    if len(results) > 0:
        bboxes = summarize_bbox(results["bbox_results"])
        _, _, locations, areas, complexities = scan_complexity(img, bboxes, grid_h, grid_w, complexity_type=complexity_type)
        assert sum(areas) == img_h*img_w, "sum(areas_sorted) = %d v.s. img_h*img_w = %d"%(sum(areas), img_h*img_w)
        if locations_pre:
            assert len(locations) == len(locations_pre)
            for loc, loc_pre in zip(locations, locations_pre):
                assert loc == loc_pre
        if areas_pre:
            assert len(areas) == len(areas_pre)
            for area, area_pre in zip(areas, areas_pre):
                assert area == area_pre
    else:
        assert (locations_pre is not None) and (areas_pre is not None) and (complexity_pre is not None)
        locations = locations_pre
        areas = areas_pre
    complexities_merged = merge_complexities(complexities=complexities, complexities_pre=complexity_pre)
    locations_sorted = [x for _, x in sorted(zip(complexities_merged, locations), key=lambda pair: pair[0])]
    areas_sorted = [x for _, x in sorted(zip(complexities_merged, areas), key=lambda pair: pair[0])]
    ratios = np.cumsum(areas_sorted) / (img_h*img_w)
    threshold = (ratios < ratio).sum()
    for i in range(threshold):
        start_h, end_h, start_w, end_w = locations_sorted[i]
        # img[:, :, start_h:end_h, start_w:end_w] = -2.1179
        img[:, 0, start_h:end_h, start_w:end_w] = -0.485/0.229
        img[:, 1, start_h:end_h, start_w:end_w] = -0.456/0.224
        img[:, 2, start_h:end_h, start_w:end_w] = -0.406/0.225
        mask[start_h:end_h, start_w:end_w] = 0

    # aa = data["img"][0].squeeze(0).permute(1,2,0).cpu().numpy()
    # aa = aa - aa.min()
    # bb = (aa / aa.max() * 255).astype(np.uint8)
    # cv2.imwrite("img_dropped_imgcomplx.png", cv2.cvtColor(bb,cv2.COLOR_RGB2BGR))
    return mask.unsqueeze(0).unsqueeze(0).cuda()


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):

            # bp()
            if data_batch['img_metas'].data[0][0]['drop_info']['meta']['ratio'] > 0:
                mask = apply_dropping(data_batch, [],
                    locations_pre=data_batch['img_metas'].data[0][0]['drop_info']['locations'],
                    areas_pre=data_batch['img_metas'].data[0][0]['drop_info']['areas'],
                    complexity_pre=data_batch['img_metas'].data[0][0]['drop_info']['complexities'],
                    grid_h=data_batch['img_metas'].data[0][0]['drop_info']['meta']['grid_h'],
                    grid_w=data_batch['img_metas'].data[0][0]['drop_info']['meta']['grid_w'],
                    ratio=data_batch['img_metas'].data[0][0]['drop_info']['meta']['ratio'],
                    complexity_type=data_batch['img_metas'].data[0][0]['drop_info']['meta']['prev_frame_complexity_type']
                )
            ########### register mask in backbone modules ##################
            # for name, module in model.named_modules():
            # for name, module in model.module.backbone.conv1.named_modules():
            #     setattr(module, "mask", mask)
            setattr(self.model.module.backbone.conv1, "mask", mask)
            for name, module in self.model.module.backbone.layer1.named_modules():
                setattr(module, "mask", mask)
            for name, module in self.model.module.backbone.layer2.named_modules():
                setattr(module, "mask", mask)
            for name, module in self.model.module.backbone.layer3.named_modules():
                setattr(module, "mask", mask)
            for name, module in self.model.module.backbone.layer4.named_modules():
                setattr(module, "mask", mask)
            ###############################################################

            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """

        ################ register hood in backbone modules to drop patch on features ####################
        self.model.module.backbone.conv1.register_forward_hook(mask_feature)
        for name, module in self.model.module.backbone.layer1.named_modules():
            module.register_forward_hook(mask_feature)
        for name, module in self.model.module.backbone.layer2.named_modules():
            module.register_forward_hook(mask_feature)
        for name, module in self.model.module.backbone.layer3.named_modules():
            module.register_forward_hook(mask_feature)
        for name, module in self.model.module.backbone.layer4.named_modules():
            module.register_forward_hook(mask_feature)
        #################################################################################################

        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)


@RUNNERS.register_module()
class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead')
        super().__init__(*args, **kwargs)
