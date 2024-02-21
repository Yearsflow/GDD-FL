import copy
import numpy as np
import torch
import random
import os
import json
import datetime
import logging
from options import get_args
from utils.common_utils import mkdirs, seed_everything, get_dataset_info
from PIL import ImageFile

if __name__ == '__main__':

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    args, extra_args = get_args()

    if args.device != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.n_classes, args.channel, args.im_size, \
    args.mean, args.std = get_dataset_info(args.dataset)

    if args.mode == 'Fed':
        if args.approach == 'fedavg':
            from Fed.fedavg import FedAvg as alg
        elif args.approach == 'fedproto':
            from Fed.fedproto import FedProto as alg
        elif args.approach == 'feddc' or args.approach == 'feddsa' or \
            args.approach == 'feddm' or args.approach == 'fedgdd':
            from Fed.feddistill import FedDistill as alg
        elif args.approach == 'fedgan':
            from Fed.fedgan import FedGAN as alg
        elif args.approach == 'feddream':
            from Fed.feddream import FedDream as alg
        elif args.approach == 'ours':
            from Fed.ours import Ours as alg
        elif args.approach == 'fedl2d':
            # from Fed.fedl2d import FedL2D as alg
            from Fed.fedl2d_data import FedL2D as alg
        else:
            raise NotImplementedError('Approach Not Implemented')
    elif args.mode == 'UpperBound':
        if args.approach == 'central':
            from UpperBound.central import Central as alg
        elif args.approach == 'rs':
            from UpperBound.rs import RS as alg
        else:
            raise NotImplementedError('Approach Not Implemented')
    else:
        raise NotImplementedError('Mode Not Supported')

    # arguments specific to the chosen FL algorithm
    extra_args = alg.extra_parser(extra_args)
    # ===================================================


    # ================ logging related ==================
    mkdirs(args.logdir)
    mkdirs(args.ckptdir)
    mkdirs(os.path.join(args.ckptdir, args.mode))
    mkdirs(os.path.join(args.ckptdir, args.mode, args.approach))

    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    argument_path = argument_path + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args) + str(extra_args), f)
    print(str(args))
    print(str(extra_args))

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    print('log path: ', log_path)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    seed_everything(args.init_seed)

    appr = alg(args, extra_args, logger)
    appr.run()