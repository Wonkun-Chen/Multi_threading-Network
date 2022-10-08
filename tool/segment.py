#coding=utf-8
import argparse
import os
import models
import time
import logging
import random
import torch
import torch.optim
cudnn.benchmark = True
import numpy as np
from data import datasets
from utils import Parser,str2bool
from predict import validate_softmax
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
parser = argparse.ArgumentParser()

parser.add_argument('-cfg', '--cfg', default='train', required=True, type=str)#setting file
parser.add_argument('-gpu', '--gpu', default='0', type=str)
parser.add_argument('-is_out', '--is_out', default=True, type=str2bool)
parser.add_argument('-verbose', '--verbose', default=True, type=str2bool)
parser.add_argument('-use_TTA', '--use_TTA', default=False, type=str2bool)
parser.add_argument('-postprocess', '--postprocess', default=True, type=str2bool)
parser.add_argument('-save_format', '--save_format', default='nii', choices=['nii','npy'], type=str)
parser.add_argument('-restore', '--restore', default=argparse.SUPPRESS, type=str)

path = os.path.dirname(__file__)

args = parser.parse_args()
args = Parser(args.cfg, log='train').add_args(args)
args.gpu = str(args.gpu)
ckpts = args.makedir()
args.resume = os.path.join(ckpts, args.restore)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    Network = getattr(models, args.net)
    model = Network(**args.net_params)
    model = torch.nn.DataParallel(model).cuda()
    assert os.path.isfile(args.resume),"no checkpoint found at {}".format(args.resume)
    checkpoint = torch.load(args.resume)
    args.start_iter = checkpoint['iter']
    model.load_state_dict(checkpoint['state_dict'])

    msg += '\n' + str(args)
    logging.info(msg)

    root_path = args.valid_data_dir
    is_scoring = False



    Dataset = getattr(datasets, args.dataset) #
    valid_list = os.path.join(root_path, args.valid_list)
    valid_set = Dataset(valid_list, root=root_path,for_train=False, transforms=args.test_transforms)

    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        collate_fn=valid_set.collate,
        num_workers=10,
        pin_memory=True)

    out_dir = './segment/{}'.format(args.cfg)
    os.makedirs(os.path.join(out_dir,'submission'),exist_ok=True)


    logging.info('-'*50)
    logging.info(msg)

    with torch.no_grad():
        validate_softmax(
            valid_loader,
            model,
            cfg=args.cfg,
            savepath=out_dir,
            save_format = args.save_format,
            names=valid_set.names,
            scoring=is_scoring,
            verbose=args.verbose,
            use_TTA=args.use_TTA,
            postprocess=args.postprocess,
            cpu_only=False)

if __name__ == '__main__':
    main()
