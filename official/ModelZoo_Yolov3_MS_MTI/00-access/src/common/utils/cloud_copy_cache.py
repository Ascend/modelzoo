import os
import shutil
import time
import argparse
import glob
from pathlib import Path

try:
    import moxing as mox
    mox.file.set_auth(is_secure=False)
except:
    print("training locally")

def _check_dir(dist_dir):
    copy_flag = True
    if os.path.exists(dist_dir):
        copy_flag = False
    if not os.path.exists(os.path.dirname(dist_dir)):
        os.makedirs(os.path.dirname(dist_dir))
    return copy_flag


def copy_pretrained_model_to_cache(src_dir='', dist_dir=''):
    start_t = time.time()
    copy_flag = _check_dir(dist_dir)

    if copy_flag:
        print('copy...')
        mox.file.copy(src_dir, dist_dir)
        print('--------------------------------------------')
        print('--dist_dir: {}'.format(dist_dir))
        print('copy pretrained model completed!')
    else:
        print("Since pretrained model already exists, copying is not required")
    end_t = time.time()
    print('copy cost time {} sec'.format(end_t - start_t))


def copy_data_to_cache(src_dir='', dist_dir=''):
    start_t = time.time()
    copy_flag = _check_dir(dist_dir)
    if copy_flag:
        print('copy {}...'.format(src_dir))
        tar_files = []
        if mox.file.is_directory(src_dir):
            mox.file.copy_parallel(src_dir, dist_dir)
        else:
            mox.file.copy(src_dir, dist_dir)
            if dist_dir.endswith('tar') or dist_dir.endswith('tar.gz'):
                tar_files.append(dist_dir)
        tar_list = list(Path(dist_dir).glob('**/*.tar'))
        tar_files.extend(tar_list)
        tar_list = list(Path(dist_dir).glob('**/*.tar.gz'))
        tar_files.extend(tar_list)
        # tar xvf tar file
        print('tar_files:{}'.format(tar_files))
        for tar_file in tar_files:
            tar_dir = os.path.dirname(tar_file)
            print('cd {}; tar -xvf {} > /dev/null 2>&1'.format(tar_dir, tar_file))
            os.system('cd {}; tar -xvf {} > /dev/null 2>&1'.format(tar_dir, tar_file))
            os.system('cd {}; rm -rf {}'.format(tar_dir, tar_file))
        print('--------------------------------------------')
        print('copy data completed!')
    else:
        print("Since data already exists, copying is not required")
    end_t = time.time()
    print('copy cost time {} sec'.format(end_t-start_t))


def copy_mox_ckpt_log_to_s3(args, ckpt):
    if args.train_url != '':
        import moxing as mox
        # change absolute dir to relative dir for os.path.join
        roma_ckpt = ckpt[1:] if ckpt[0] == '/' else ckpt
        roma_weights_fp = os.path.join(args.train_url, roma_ckpt)
        roma_weights_dirname = os.path.dirname(roma_weights_fp)
        if not mox.file.exists(roma_weights_dirname):
            mox.file.make_dirs(roma_weights_dirname)
        os.system("python -c 'import moxing as mox; mox.file.copy(\"{}\", \"{}\")' &".format(ckpt, roma_weights_fp))
        args.logger.info("save weight success, local_weights_fp:{}, roma_weights_fp:{}".format(ckpt, roma_weights_fp))

        # change absolute dir to relative dir for os.path.join
        roma_log_fn = args.logger.log_fn[1:] if args.logger.log_fn[0] == '/' else args.logger.log_fn
        roma_log_fp = os.path.join(args.train_url, roma_log_fn)
        roma_log_dirname = os.path.dirname(roma_log_fp)
        if not mox.file.exists(roma_log_dirname):
            mox.file.make_dirs(roma_log_dirname)
        os.system("python -c 'import moxing as mox; mox.file.copy(\"{}\", \"{}\")' &".format(args.logger.log_fn, roma_log_fp))
        args.logger.info("save log success, local_log_fp:{}, roma_log_fp:{}".format(args.logger.log_fn, roma_log_fp))
