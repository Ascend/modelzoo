import mindspore.nn as nn

def mixed_precision_warpper(network, opt_level='O2', OP_fp32=None):
    # note: the first of opt_level is o, not zero
    def _find_match_OP(network, fp32_ops):
        for name, cell in network.cells_and_names():
            if isinstance(cell, fp32_ops):
                cell.add_flags_recursive(fp32=True)
        network.add_flags_recursive(fp16=True)

    if opt_level=='O0':
        network.add_flags_recursive(fp32=True)
    elif opt_level=='O1':
        if not OP_fp32:
            raise Exception('choose options to set as fp32')
        _find_match_OP(network, OP_fp32)
    elif opt_level=='O2':
        _find_match_OP(network, (nn.BatchNorm2d))
    elif opt_level=='O3':
        # not O3 can cause unknown problems, training will be extremely slow
        network.add_flags_recursive(fp16=True)
    else:
        raise NotImplementedError('choose opt_level from O0 to O3')
