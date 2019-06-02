import argparse
import pprint
import numpy as np
import pdb
import sys
import os.path

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')
# for p in sys.path: print p
# print (this_dir)

from function_network.zlrm.solverwrapper.fpn_solverwrapper_op_4step import get_training_roidb, train_net
from lib.networks.netconfig import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='4 op Train a networks')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=3000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default='D:\\jjj\\zlrm\\data\\zlrm_data\\parameter\\Resnet50.npy', type=str)
    parser.add_argument('--fpn_rpn_output', dest='fpn_rpn_output',
                        help='fpn rpn ckpt 输出路径',
                        default='D:\\jjj\\zlrm\\output\\fpn_rpn', type=str)
    parser.add_argument('--fpn_detect_output', dest='fpn_detect_output',
                        help='fpn detect ckpt 输出路径',
                        default='D:\\jjj\\zlrm\\output\\fpn_detect', type=str)
    parser.add_argument('--fpn_shared_conv_rpn_output', dest='fpn_shared_conv_rpn_output',
                        help='fpn shared conv rpn network ckpt 输出路径',
                        default='D:\\jjj\\zlrm\\output\\fpn_shared_conv_rpn_network', type=str)
    parser.add_argument('--fpn_shared_conv_detect_output', dest='fpn_shared_conv_detect_output',
                        help='fpn shared conv detect network ckpt 输出路径',
                        default='D:\\jjj\\zlrm\\output\\fpn_shared_conv_detect_network', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='zlrm_data_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--fpn_rpn_networks', dest='fpn_rpn_network_name',
                        help='name of the networks',
                        default='FPN_Resnet50_train_rpn', type=str)
    parser.add_argument('--fpn_detect_networks', dest='fpn_detect_network_name',
                        help='name of the networks',
                        default='FPN_Resnet50_train_detect', type=str)
    parser.add_argument('--fpn_shared_conv_networks', dest='fpn_shared_conv_network_name',
                        help='name of the networks',
                        default='FPN_Resnet50_train_shared_conv', type=str)
    parser.add_argument('--train_step', dest='train_step',
                        help='训练的网络部分，四步：'
                             'train_step_rpn;train_step_detect;train_step_shared_conv_rpn;train_step_shared_conv_detect',
                        default='train_step_detect', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--restore', dest='restore',
                        help='restore or not',
                        default=1, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        # sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.ZLRM.RNG_SEED)
    imdb = get_imdb(args.imdb_name)
    print ('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    device_name = '/gpu:{:d}'.format(args.gpu_id)
    print (device_name)

    train_net(args.fpn_rpn_network_name, args.fpn_detect_network_name, args.fpn_shared_conv_network_name, imdb, roidb,
              rpn_output_dir=args.fpn_rpn_output,
              detect_output_dir=args.fpn_detect_output,
              shared_conv_rpn_output_dir=args.fpn_shared_conv_rpn_output,
              shared_conv_detect_output_dir=args.fpn_shared_conv_detect_output,
              train_step=args.train_step,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters,
              restore=bool(int(args.restore)))
