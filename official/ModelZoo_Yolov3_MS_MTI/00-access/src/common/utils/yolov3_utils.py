from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.nn as nn


def load_backbone(net, ckpt_path, args):
    param_dict = load_checkpoint(ckpt_path)
    # net = yolov3_darknet53()
    yolo_backbone_prefix = 'feature_map.backbone'
    darknet_backbone_prefix = 'network.backbone'
    find_param = []
    not_found_param = []

    for name, cell in net.cells_and_names():
        if name.startswith(yolo_backbone_prefix):
            # fw_yolo_param.write(name+'\n')
            name = name.replace(yolo_backbone_prefix, darknet_backbone_prefix)
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                darknet_weight = '{}.weight'.format(name)
                darknet_bias = '{}.bias'.format(name)
                if darknet_weight in param_dict:
                    cell.weight.default_input = param_dict[darknet_weight].data
                    find_param.append(darknet_weight)
                else:
                    not_found_param.append(darknet_weight)
                if darknet_bias in param_dict:
                    cell.bias.default_input = param_dict[darknet_bias].data
                    find_param.append(darknet_bias)
                else:
                    not_found_param.append(darknet_bias)
            elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
                darknet_moving_mean      = '{}.moving_mean'.format(name)
                darknet_moving_variance  = '{}.moving_variance'.format(name)
                darknet_gamma            = '{}.gamma'.format(name)
                darknet_beta             = '{}.beta'.format(name)
                if darknet_moving_mean in param_dict:
                    cell.moving_mean.default_input = param_dict[darknet_moving_mean].data
                    find_param.append(darknet_moving_mean)
                else:
                    not_found_param.append(darknet_moving_mean)
                if darknet_moving_variance in param_dict:
                    cell.moving_variance.default_input = param_dict[darknet_moving_variance].data
                    find_param.append(darknet_moving_variance)
                else:
                    not_found_param.append(darknet_moving_variance)
                if darknet_gamma in param_dict:
                    cell.gamma.default_input = param_dict[darknet_gamma].data
                    find_param.append(darknet_gamma)
                else:
                    not_found_param.append(darknet_gamma)
                if darknet_beta in param_dict:
                    cell.beta.default_input = param_dict[darknet_beta].data
                    find_param.append(darknet_beta)
                else:
                    not_found_param.append(darknet_beta)

    # fw_yolo_param.close()
    args.logger.info('================found_param {}========='.format(len(find_param)))
    args.logger.info(find_param)
    args.logger.info('================not_found_param {}========='.format(len(not_found_param)))
    args.logger.info(not_found_param)
    args.logger.info('=====load {} successfully ====='.format(ckpt_path))

    return net
