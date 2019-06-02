from lib.generate_defect.zlrm_Generate_the_defects_data import Generate_Defect
import os
import argparse
import sys
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='defect')
    parser.add_argument('--save_image_root', dest='save_image_root',
                        help='save_image_root',
                        default='D:\\generate_zlrm\\save_image', type=str)
    parser.add_argument('--save_label_root', dest='save_label_root',
                        help='save_label_root',
                        default='D:\\generate_zlrm\\save_label', type=str)
    parser.add_argument('--read_label_root', dest='read_label_root',
                        help='read_label_root',
                        default='D:\\generate_zlrm\\label', type=str)
    parser.add_argument('--read_defect_root', dest='read_defect_root',
                        help='read_defect_root',
                        default='D:\\generate_zlrm\\defect', type=str)
    parser.add_argument('--preload_defect_root', dest='preload_defect_root',
                        help='preload_defect_root',
                        default='D:\\generate_zlrm\\preload_defect', type=str)
    parser.add_argument('--save_fail_label_root', dest='save_fail_label_root',
                        help='save_fail_label_root',
                        default='D:\\generate_zlrm\\fail_label', type=str)
    parser.add_argument('--save_fail_image_root', dest='save_fail_image_root',
                        help='save_fail_image_root',
                        default='D:\\generate_zlrm\\fail_image', type=str)
    parser.add_argument('--read_image_root', dest='read_image_root',
                        help='read_image_root',
                        default='D:\\jjj\\zlrm\\data\\zlrm_data\\datasets\\ImageSets', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        # sys.exit(1)

    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    gd = Generate_Defect(save_image_root=args.save_image_root,
                         save_label_root=args.save_label_root,
                         read_label_root=args.read_label_root,
                         read_defect_root=args.read_defect_root,
                         save_fail_label_root= args.save_fail_label_root,
                         save_fail_image_root= args.save_fail_image_root)

    # 预处理小的缺陷图片，将纹理部分全部设为纯白
    # gd.preload_defect(args.preload_defect_root)
    # 将缺陷与铝锭图片合成
    gd.generate_defect_batch(args.read_image_root)