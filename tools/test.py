import argparse
import os
import sys
from pathlib import Path
lib_dir = (Path(__file__).parent / '..').resolve()
sys.path.insert(0, str(lib_dir))

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
# from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataset
import torch.nn.utils.prune as prune
import torch.nn as nn
from qdtrack.core.utils import fp_quantization

from pdb import set_trace as st


def prune_model_l1_unstructured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.l1_unstructured(module, 'weight', proportion)
            prune.remove(module, 'weight')
    return model

def prune_model_global_unstructured(model, layer_type, proportion):
    module_tups = []
    for module in model.modules():
        if isinstance(module, layer_type):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='qdtrack test model')
    parser.add_argument(
        '--prune', default=0.0, type=float, help='portion of network to be pruned')  # 0.7~
    parser.add_argument('--prune_method', default='global', type=str,
                        help="layer-wise / global ",
                        choices=['layer', 'global'])
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--vanilla_test',
        action='store_true',
        help='original qdtrack testing')
    parser.add_argument(
        '--compose_type',
        type=str,
        default='bottom_up',
        help='how to combine the two complexities')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument('--eval', type=str, nargs='+', help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--show_score_thr', default=0.3, type=float, help='output result file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--quant_weight', action='store_true', default=False)
    parser.add_argument('--quant_act', action='store_true', default=False)
    parser.add_argument('--quantization', default='b', type=str,
                        help="balanced / nonbalanced",
                        choices=['b', 'nb', 'fg', 'lfp'])
    parser.add_argument(
        '--q', default=8, type=int, help='quantization number {b: 7, 15; nb: 8, 16}')
    parser.add_argument(
        '--int', default=3, type=int, help='quantization number {1}')
    parser.add_argument(
        '--dec', default=12, type=int, help='quantization number {6， 14}')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('USE_MMDET', False):
        from mmdet.apis import multi_gpu_test, single_gpu_test
        from mmdet.models import build_detector as build_model
        from mmdet.datasets import build_dataloader
    else:
        from qdtrack.apis import multi_gpu_test, single_gpu_test, single_gpu_test_vanilla
        from qdtrack.models import build_model
        from qdtrack.datasets import build_dataloader

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=None)
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if args.quant_weight:
        # Quantization
        if args.quantization == 'b':
            print(f'> Quantizing model weight.....')
            quantization = 2**args.q -1
            quantDict = {}
            for keys in model.state_dict().keys():
                quantDict[keys] = {}
                quantDict[keys]['max'] = torch.max(model.state_dict()[keys])
                quantDict[keys]['min'] = torch.min(model.state_dict()[keys])

            state_dict = model.state_dict()
            for keys in state_dict.keys():
                normValue = max(torch.abs(quantDict[keys]['max']), torch.abs(quantDict[keys]['min']))
                state_dict[keys] = torch.round(state_dict[keys] * quantization / normValue) / quantization * normValue
            model.load_state_dict(state_dict)
        elif args.quantization == 'fg':
            state_dict = model.state_dict()
            for keys in state_dict.keys():
                state_dict[keys] = fp_quantization(state_dict[keys], args.int, args.dec)
            model.load_state_dict(state_dict)

        elif args.quantization == 'lfp':
            state_dict = model.state_dict()
            bit_choice = []
            for keys in state_dict.keys():
                temp_max = torch.max(model.state_dict()[keys])
                temp_min = torch.min(model.state_dict()[keys])

                temp = temp_max

                int_bit = 0
                a = 999
                while a != 0:
                    a = temp // 2
                    if a > 0:
                        int_bit = int_bit + 1
                    temp = temp / 2.0
                dec_bit = args.q - 1 - int_bit

                bit_choice += [{"Params":keys, "max": temp_max, "min":temp_min, "int_bit":int_bit, "dec_bit":dec_bit}]
                state_dict[keys] = fp_quantization(state_dict[keys], int_bit, dec_bit)

    # Pruning
    if args.prune_method == 'layer':
        model = prune_model_l1_unstructured(model, nn.Conv1d, args.prune)
        model = prune_model_l1_unstructured(model, nn.Conv2d, args.prune)
        model = prune_model_l1_unstructured(model, nn.Conv3d, args.prune)
    elif args.prune_method == 'global':
        model = prune_model_global_unstructured(model, (nn.Conv1d, nn.Conv2d, nn.Conv3d), args.prune)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        if args.vanilla_test:
            outputs = single_gpu_test_vanilla(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
        else:
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                    args.show_score_thr, args.compose_type, args=args)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
