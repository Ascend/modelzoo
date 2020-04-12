
def default_wd_filter(x):
    # default weight decay filter
    parameter_name = x.name
    if parameter_name.endswith('bias'):
        # all bias not using weight decay
        return False
    elif 'bn' in parameter_name:
        # bn weight bias not using weight decay, be carefully for now x not include bn
        return False
    else:
        return True
