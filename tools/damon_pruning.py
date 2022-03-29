import torch.nn.utils.prune as prune
from collections.abc import Iterable
import torch
import torch.nn as nn
import operator

def global_custom_unstructured(parameters, pruning_method, sparsity_threshold = 9, pattern_num=0, importance_scores=None,  **kwargs):
    r"""
    Globally prunes tensors corresponding to all parameters in ``parameters``
    by applying the specified ``pruning_method``.
    Modifies modules in place by:
    1) adding a named buffer called ``name+'_mask'`` corresponding to the
    binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    ``name+'_orig'``.

    Args:
        parameters (Iterable of (module, name) tuples): parameters of
            the model to prune in a global fashion, i.e. by aggregating all
            weights prior to deciding which ones to prune. module must be of
            type :class:`nn.Module`, and name must be a string.
        pruning_method (function): a valid pruning function from this module,
            or a custom one implemented by the user that satisfies the
            implementation guidelines and has ``PRUNING_TYPE='unstructured'``.
        importance_scores (dict): a dictionary mapping (module, name) tuples to
            the corresponding parameter's importance scores tensor. The tensor
            should be the same shape as the parameter, and is used for computing
            mask for pruning.
            If unspecified or None, the parameter will be used in place of its
            importance scores.
        kwargs: other keyword arguments such as:
            amount (int or float): quantity of parameters to prune across the
            specified parameters.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Raises:
        TypeError: if ``PRUNING_TYPE != 'unstructured'``

    Note:
        Since global structured pruning doesn't make much sense unless the
        norm is normalized by the size of the parameter, we now limit the
        scope of global pruning to unstructured methods.

    Examples:
        >>> net = nn.Sequential(OrderedDict([
                ('first', nn.Linear(10, 4)),
                ('second', nn.Linear(4, 1)),
            ]))
        >>> parameters_to_prune = (
                (net.first, 'weight'),
                (net.second, 'weight'),
            )
        >>> prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=10,
            )
        >>> print(sum(torch.nn.utils.parameters_to_vector(net.buffers()) == 0))
        tensor(10, dtype=torch.uint8)

    """

    """pattern WM mobile"""
    pattern1 = [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]]

    pattern2 = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 0]]
    pattern3 = [[0, 0], [0, 1], [0, 2], [2, 1], [2, 2]]
    pattern4 = [[0, 2], [1, 0], [1, 2], [2, 0], [2, 2]]
    pattern5 = [[0, 0], [0, 1], [2, 0], [2, 1], [2, 2]]
    pattern6 = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 2]]
    pattern7 = [[0, 1], [0, 2], [2, 0], [2, 1], [2, 2]]
    pattern8 = [[0, 0], [1, 0], [1, 2], [2, 0], [2, 2]]
    pattern9 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1]]

    pattern10 = [[0, 1], [0, 2], [1, 2], [2, 0], [2, 2]]
    pattern11 = [[0, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
    pattern12 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 1]]
    pattern13 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 2]]
    pattern14 = [[0, 0], [0, 1], [1, 0], [2, 0], [2, 2]]
    pattern15 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]
    pattern16 = [[0, 0], [0, 2], [1, 2], [2, 1], [2, 2]]
    pattern17 = [[0, 2], [1, 0], [2, 0], [2, 1], [2, 2]]

    pattern18 = [[0, 0], [0, 2], [2, 0], [2, 1], [2, 2]]
    pattern19 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]
    pattern20 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]
    pattern21 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]

    patterns_dict = {1: pattern1,
                     2: pattern2,
                     3: pattern3,
                     4: pattern4,
                     5: pattern5,
                     6: pattern6,
                     7: pattern7,
                     8: pattern8,
                     9: pattern9,
                     10: pattern10,
                     11: pattern11,
                     12: pattern12,
                     13: pattern13,
                     14: pattern14,
                     15: pattern15,
                     16: pattern16,
                     17: pattern17,
                     18: pattern18,
                     19: pattern19,
                     20: pattern20,
                     21: pattern21
                     }


    # ensure parameters is a list or generator of tuples
    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")

    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")

    # flatten importance scores to consider them all at once in global pruning
    relevant_importance_scores = torch.nn.utils.parameters_to_vector(
        [
            importance_scores.get((module, name), getattr(module, name))
            for (module, name) in parameters
        ]
    )
    # similarly, flatten the masks (if they exist), or use a flattened vector
    # of 1s of the same dimensions as t
    default_mask = torch.nn.utils.parameters_to_vector(
        [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
        ]
    )

    # use the canonical pruning methods to compute the new mask, even if the
    # parameter is now a flattened out version of `parameters`
    container = prune.PruningContainer()
    container._tensor_name = "temp"  # to make it match that of `method`
    method = pruning_method(**kwargs)
    method._tensor_name = "temp"  # to make it match that of `container`
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for '
            "the `pruning_method`. Found method {} of type {}".format(
                pruning_method, method.PRUNING_TYPE
            )
        )

    container.add_pruning_method(method)

    # use the `compute_mask` method from `PruningContainer` to combine the
    # mask computed by the new method with the pre-existing mask
    final_mask = container.compute_mask(relevant_importance_scores, default_mask)

    # Pointer for slicing the mask to match the shape of each parameter
    pointer = 0
    custom_pointer = 0

    for module, name in parameters:

        # Modify mask according to damon's rule.
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            (in_channel, out_channel, w, h) = module.weight.shape

            if w == 3 and h == 3:

                if pattern_num == 0:
                    for a in range(in_channel):
                        for c in range(out_channel):
                            sparsity_count = 0
                            for m in range(9):
                                if final_mask[custom_pointer + m] == 0:
                                    sparsity_count = sparsity_count + 1

                            if sparsity_count == 9:
                                custom_pointer = custom_pointer + 9

                            elif sparsity_count >= sparsity_threshold:
                                final_mask[custom_pointer:custom_pointer+9] = 0
                                custom_pointer = custom_pointer + 9

                            else:
                                final_mask[custom_pointer:custom_pointer + 9] = 1
                                custom_pointer = custom_pointer + 9

                else:
                    for a in range(in_channel):
                        for c in range(out_channel):
                            sparsity_count = 0
                            for m in range(9):
                                if final_mask[custom_pointer + m] == 0:
                                    sparsity_count = sparsity_count + 1

                            if sparsity_count == 9:
                                custom_pointer = custom_pointer + 9

                            elif sparsity_count >= pattern_num:
                                final_mask[custom_pointer:custom_pointer + 9] = 1

                                with torch.no_grad():
                                    current_kernel = module.weight[a, c, :, :]
                                    temp_dict = {}  # store each pattern's norm value on the same weight kernel

                                    # Pattern Search
                                    for key, pattern in patterns_dict.items():
                                        temp_kernel = current_kernel
                                        for index in pattern:
                                            temp_kernel[index[0], index[1]] = 0
                                        current_norm = torch.norm(temp_kernel)
                                        temp_dict[key] = current_norm

                                    best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]


                                #Create mask
                                for index in patterns_dict[best_pattern]:
                                    final_mask[custom_pointer+ index[0]*3 + index[1] +1] = 0

                                custom_pointer = custom_pointer + 9

                            else:
                                final_mask[custom_pointer:custom_pointer + 9] = 1
                                custom_pointer = custom_pointer + 9

        else:
            custom_param = getattr(module, name)
            num_custom_param = custom_param.numel()
            custom_pointer += num_custom_param

        param = getattr(module, name)
        # The length of the parameter
        num_param = param.numel()
        # Slice the mask, reshape it
        param_mask = final_mask[pointer : pointer + num_param].view_as(param)
        # Assign the correct pre-computed mask to each parameter and add it
        # to the forward_pre_hooks like any other pruning method
        prune.custom_from_mask(module, name, mask=param_mask)

        # Increment the pointer to continue slicing the final_mask
        pointer += num_param
