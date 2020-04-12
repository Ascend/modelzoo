
def default_wd_filter(x):
    # default weight decay filter
    parameter_name = x.name
    if parameter_name.endswith('.bias'):
        # all bias not using weight decay
        # print('no decay:{}'.format(parameter_name))
        return False
    elif parameter_name.endswith('.gamma'):
        # bn weight bias not using weight decay, be carefully for now x not include BN
        # print('no decay:{}'.format(parameter_name))
        return False
    elif parameter_name.endswith('.beta'):
        # bn weight bias not using weight decay, be carefully for now x not include BN
        # print('no decay:{}'.format(parameter_name))
        return False
    else:
        return True
