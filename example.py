import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import OrderedDict

import torch
import torchvision.utils as vutils

import utils
from models import DGP

sys.path.append("./")


# Arguments for demo
def add_example_parser(parser):
    parser.add_argument(
        '--image_path', type=str, default='',
        help='Path of the image to be processed (default: %(default)s)')
    parser.add_argument(
        '--class', type=int, default=-1,
        help='class index of the image (default: %(default)s)')
    parser.add_argument(
        '--image_path2', type=str, default='',
        help='Path of the 2nd image to be processed, used in "morphing" mode (default: %(default)s)')
    parser.add_argument(
        '--class2', type=int, default=-1,
        help='class index of the 2nd image, used in "morphing" mode (default: %(default)s)')
    return parser


# prepare arguments and save in config
parser = utils.prepare_parser()
parser = utils.add_dgp_parser(parser)
parser = add_example_parser(parser)
config = vars(parser.parse_args())
utils.dgp_update_config(config)

# set random seed
utils.seed_rng(config['seed'])

if not os.path.exists('{}/images'.format(config['exp_path'])):
    os.makedirs('{}/images'.format(config['exp_path']))
if not os.path.exists('{}/images_sheet'.format(config['exp_path'])):
    os.makedirs('{}/images_sheet'.format(config['exp_path']))

# initialize DGP model
dgp = DGP(config)

# prepare the target image
img = utils.get_img(config['image_path'], config['resolution']).cuda()
category = torch.Tensor([config['class']]).long().cuda()
dgp.set_target(img, category, config['image_path'])

# prepare initial latent vector
dgp.select_z(select_y=True if config['class'] < 0 else False)
# start reconstruction
loss_dict = dgp.run()

