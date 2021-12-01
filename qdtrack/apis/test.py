import os.path as osp
import shutil
import tempfile
import time
from collections import defaultdict

import numpy as np
import mmcv
from mmdet.models.builder import BACKBONES
import torch
from torch import nn
import torch.distributed as dist
from mmcv.runner import get_dist_info
from pdb import set_trace as bp


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


from torchvision import transforms
inv_normalizer = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
BACKGROUND = -2.1179


THRESHOLD = 0.01 # detection confidence
RATIO = 0.2
GRID_H = GRID_W = 60
from shapely.geometry import box
def box_iou(loc, bboxes):
    rect_target = box(*loc)
    area = 0
    for rect in bboxes:
        # https://stackoverflow.com/questions/39049929/finding-the-area-of-intersection-of-multiple-overlapping-rectangles-in-python
        area += rect_target.intersection(rect).area
    return area


# detection prediction from frame t-1 as complexity
def scan_complexity(image, bboxes, grid_h, grid_w):
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
            complexities.append(box_iou([start_w, start_h, end_w, end_h], bboxes))
            # complexities.append(box_iou([start_h, start_w, end_h, end_w], bboxes))
            # complexities.append(box_iou([start_w, end_h, end_w, start_h], bboxes))
            locations.append([start_h, end_h, start_w, end_w])
            areas.append((end_h-start_h)*(end_w-start_w))
            start_w += grid_w
        start_h += grid_h
    locations_sorted = [x for _, x in sorted(zip(complexities, locations))]
    areas_sorted = [x for _, x in sorted(zip(complexities, areas))]
    return locations_sorted, areas_sorted, locations, areas, complexities


def bbox_zeros(image, bboxes, ratio, grid_h, grid_w):
    img_h, img_w = image.shape[2:]
    locations_sorted, areas_sorted, locations, areas, complexities = scan_complexity(image, bboxes, grid_h, grid_w)
    assert sum(areas_sorted) == img_h*img_w, "sum(areas_sorted) = %d v.s. img_h*img_w = %d"%(sum(areas_sorted), img_h*img_w)
    ratios = np.cumsum(areas_sorted) / (img_h*img_w)
    threshold = (ratios < ratio).sum()
    for i in range(threshold):
        start_h, end_h, start_w, end_w = locations_sorted[i]
        image[:, :, start_h:end_h, start_w:end_w] = BACKGROUND
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


def drop_by_bbox(data, results, grid_h=GRID_H, grid_w=GRID_W, ratio=RATIO):
    import time
    if len(results) > 0:
        start_time = time.time()
        bboxes = summarize_bbox(results["bbox_results"])
        # print("summarize_bbox", time.time() - start_time)
        # only support batch_size = 1 for now, since bbox is only from one frame
        # data["img"][0] is still BCHW, B = 1
        start_time = time.time()
        data["img"][0] = bbox_zeros(data["img"][0], bboxes, ratio, grid_h, grid_w)
        # print("bbox_zeros", time.time() - start_time)
    return data


def merge_complexities(complexities=None, complexities_pre=None):
    complexities
    return


def apply_dropping(data, results, locations_pre=None, areas_pre=None, complexity_pre=None, grid_h=GRID_H, grid_w=GRID_W, ratio=RATIO):
    """
    data: dict of 'img' and 'img_metas' for current frame
    results: predictions from last frame
    locations_pre: patches cropped during preprocessing in dataloader
    areas_pre: areas of patches cropped during preprocessing in dataloader
    complexity_pre: complexities of patches from preprocessing in dataloader (by measuring nature image complexity)
    """
    img = data["img"][0]
    img_h, img_w = img.shape[2:]
    complexities = None
    if len(results) > 0:
        bboxes = summarize_bbox(results["bbox_results"])
        _, _, locations, areas, complexities = scan_complexity(img, bboxes, grid_h, grid_w)
        bp()
        assert sum(areas) == img_h*img_w, "sum(areas_sorted) = %d v.s. img_h*img_w = %d"%(sum(areas), img_h*img_w)
        if locations_pre:
            for loc, loc_pre in zip(locations, locations_pre):
                assert loc == loc_pre
        if areas_pre:
            for area, area_pre in zip(areas, areas_pre):
                assert area == area_pre
    complexities_merged = merge_complexities(complexities=complexities, complexities_pre=complexity_pre)
    locations_sorted = [x for _, x in sorted(zip(complexities_merged, locations))]
    areas_sorted = [x for _, x in sorted(zip(complexities_merged, areas))]
    ratios = np.cumsum(areas_sorted) / (img_h*img_w)
    threshold = (ratios < ratio).sum()
    for i in range(threshold):
        start_h, end_h, start_w, end_w = locations_sorted[i]
        img[:, :, start_h:end_h, start_w:end_w] = -2.1179
    return


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()

    # features_collector0 = Features()
    # features_collector1 = Features()
    # features_collector2 = Features()
    # features_collector3 = Features()
    # features_collector4 = Features()
    # model.module.backbone.conv1.register_forward_hook(features_collector0)
    # model.module.backbone.layer1[-1].register_forward_hook(features_collector1)
    # model.module.backbone.layer2[-1].register_forward_hook(features_collector2)
    # model.module.backbone.layer3[-1].register_forward_hook(features_collector3)
    # model.module.backbone.layer4[-1].register_forward_hook(features_collector4)

    result = defaultdict(list) # output of each single step
    results = defaultdict(list)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        # data = drop_by_bbox(data, result) # drop by bbox predicted from t-1 frame
        bp()
        apply_dropping(data, result, locations_pre=data['img_metas']['locations'], areas_pre=data['img_metas']['areas'], complexity_pre=data['img_metas']['complexities'], grid_h=data['img_metas']['grid_h'], grid_w=data['img_metas']['grid_w'], ratio=data['img_metas']['ratio'])

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # n, c, h, w = features_collector.outputs[-1].shape
        # attention = torch.einsum('nchw,nc->nhw', [features_collector.outputs[-1], nn.functional.adaptive_avg_pool2d(features_collector.outputs[-1], (1, 1)).view(1, c)])
        # attention = attention / attention.view(n, -1).sum(1).view(n, 1, 1).repeat(1, h, w)
        # np.save("/home/chenwy/qdtrack/attention.npy", attention.detach().cpu().numpy())
        # print(data['img_metas'][0]._data[0][0]['filename'])

        # from pytorch_grad_cam import GradCAM
        # from pytorch_grad_cam.utils.image import show_cam_on_image
        # model.module.backbone.out_indices = (3,)
        # cam = GradCAM(model=model.module.backbone, target_layer=model.module.backbone.layer4[-1], use_cuda=True)
        # grayscale_cam = cam(input_tensor=data['img'][0])
        # result = model(return_loss=False, rescale=True, **data)

        for k, v in result.items():
            results[k].append(v)

        if show or out_dir:
            pass  # TODO

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = defaultdict(list)
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        for k, v in result.items():
            results[k].append(v)

        if rank == 0:
            batch_size = (
                len(data['img_meta']._data)
                if 'img_meta' in data else data['img'][0].size(0))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        raise NotImplementedError
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = defaultdict(list)
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_file = mmcv.load(part_file)
            for k, v in part_file.items():
                part_list[k].extend(v)
        shutil.rmtree(tmpdir)
        return part_list
