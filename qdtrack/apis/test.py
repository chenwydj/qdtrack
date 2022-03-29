import os.path as osp
import pdb
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
from pdb import set_trace as st
import cv2

def single_gpu_test_vanilla(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = defaultdict(list)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        for k, v in result.items():
            results[k].append(v)

        if show or out_dir:
            pass  # TODO

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def save_fig(data, save_path="1.png"):
    if "img" in data:
        res = data["img"][0].squeeze(0).permute(1,2,0).cpu().numpy()
        res = res - res.min()
        res = (res / res.max() * 255).astype(np.uint8)
    else:
        res = data.squeeze(0).permute(1,2,0).cpu().numpy()
        res = res - res.min()
        res = (res / res.max() * 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

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


def merge_complexities(complexities=None, complexities_pre=None, merge=True, norm=True, comp_type="multiply"):
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


def apply_dropping(data, results, locations_pre=None, areas_pre=None, complexity_pre=None, grid_h=GRID_H, grid_w=GRID_W, ratio=RATIO, 
    complexity_type="intersection", compose_type='bottom_up'):
    """
    data: dict of 'img' and 'img_metas' for current frame
    results: predictions from last frame
    locations_pre: patches cropped during preprocessing in dataloader
    areas_pre: areas of patches cropped during preprocessing in dataloader
    complexity_pre: complexities of patches from preprocessing in dataloader (by measuring nature image complexity)
    """
    assert compose_type in ['bottom_up', 'mean', 'multiply']

    img = data["img"][0]
    mask = torch.ones(img.shape[2], img.shape[3]).to(img.device)
    # aa = data["img"][0].squeeze(0).permute(1,2,0).cpu().numpy()
    # aa = aa - aa.min()
    # bb = (aa / aa.max() * 255).astype(np.uint8)
    # cv2.imwrite("img_ori.png", cv2.cvtColor(bb,cv2.COLOR_RGB2BGR))

    img_h, img_w = img.shape[2:]
    complexities = None
    if compose_type in ['mean', 'multiply']:
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
    elif compose_type in ['bottom_up']:
        complexities_merged = complexity_pre
        locations = locations_pre
        areas = areas_pre
    else:
        raise NotImplementedError

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
    return mask.unsqueeze(0).unsqueeze(0).cuda()


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3, 
                    compose_type='bottom_up'):
    model.eval()
    
    print(f"[Compose Type]: {compose_type}")
    ################ register hood in backbone modules to drop patch on features ####################
    model.module.backbone.conv1.register_forward_hook(mask_feature)
    for name, module in model.module.backbone.layer1.named_modules():
        module.register_forward_hook(mask_feature)
    for name, module in model.module.backbone.layer2.named_modules():
        module.register_forward_hook(mask_feature)
    for name, module in model.module.backbone.layer3.named_modules():
        module.register_forward_hook(mask_feature)
    for name, module in model.module.backbone.layer4.named_modules():
        module.register_forward_hook(mask_feature)
    #################################################################################################

    # features_collector0 = Features()
    # features_collector1 = Features()
    # features_collector2 = Features()
    # features_collector3 = Features()
    # features_collector4 = Features()
    # model.module.backbone.conv1.register_forward_hook(features_collector0)
    # model.module.backbone.layer1[-2].register_forward_hook(features_collector1)
    # model.module.backbone.layer2[-2].register_forward_hook(features_collector2)
    # model.module.backbone.layer3[-2].register_forward_hook(features_collector3)
    # model.module.backbone.layer4[-2].register_forward_hook(features_collector4)

    result = defaultdict(list) # output of each single step
    results = defaultdict(list)
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        # data = drop_by_bbox(data, result) # drop by bbox predicted from t-1 frame
        if data['img_metas'][0].data[0][0]['drop_info']['meta']['ratio'] > 0:
            mask = apply_dropping(data, result,
                locations_pre=data['img_metas'][0].data[0][0]['drop_info']['locations'],
                areas_pre=data['img_metas'][0].data[0][0]['drop_info']['areas'],
                complexity_pre=data['img_metas'][0].data[0][0]['drop_info']['complexities'],
                grid_h=data['img_metas'][0].data[0][0]['drop_info']['meta']['grid_h'],
                grid_w=data['img_metas'][0].data[0][0]['drop_info']['meta']['grid_w'],
                ratio=data['img_metas'][0].data[0][0]['drop_info']['meta']['ratio'],
                complexity_type=data['img_metas'][0].data[0][0]['drop_info']['meta']['prev_frame_complexity_type'],
                compose_type=compose_type
            )
        ########### register mask in backbone modules ##################
        # for name, module in model.named_modules():
        # for name, module in model.module.backbone.conv1.named_modules():
        #     setattr(module, "mask", mask)
        setattr(model.module.backbone.conv1, "mask", mask)
        for name, module in model.module.backbone.layer1.named_modules():
            setattr(module, "mask", mask)
        for name, module in model.module.backbone.layer2.named_modules():
            setattr(module, "mask", mask)
        for name, module in model.module.backbone.layer3.named_modules():
            setattr(module, "mask", mask)
        for name, module in model.module.backbone.layer4.named_modules():
            setattr(module, "mask", mask)
        ###############################################################

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        # feat0 = features_collector0.outputs[0].detach().cpu().numpy()
        # feat1 = features_collector1.outputs[0].detach().cpu().numpy()
        # feat2 = features_collector2.outputs[0].detach().cpu().numpy()
        # feat3 = features_collector3.outputs[0].detach().cpu().numpy()
        # feat4 = features_collector4.outputs[0].detach().cpu().numpy()
        # np.save("/home/zhiwen/projects/qdtrack/work_dirs/vis/feat0.npy", feat0)
        # np.save("/home/zhiwen/projects/qdtrack/work_dirs/vis/feat1.npy", feat1)
        # np.save("/home/zhiwen/projects/qdtrack/work_dirs/vis/feat2.npy", feat2)
        # np.save("/home/zhiwen/projects/qdtrack/work_dirs/vis/feat3.npy", feat3)
        # np.save("/home/zhiwen/projects/qdtrack/work_dirs/vis/feat4.npy", feat4)
        # st()

        # if i == 10:
        #     save_fig(data, "img_dropped_compMulti.png")
        #     exit(0)
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
