from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
import scipy.io
from metrics import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset_loader import OmniDepthDataset
import cv2
import spherical as S360
import supervision as L
from util import *
#import weight_init
from sync_batchnorm import convert_model
from CasStereoNet.models import psmnet_spherical_up_down_inv_depth
from CasStereoNet.models.loss import stereo_psmnet_loss
from network_resnet_v2 import ResNet360
#from network_resnet_v2 import ResNet360
# network_rectnet import RectNet
import matplotlib.pyplot as plot
import scipy.io
import csv


## Omnifusion:
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
from torch.nn.modules.module import register_module_full_backward_hook
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import math
from metrics import *
from tqdm import tqdm

from _extra.OmniFusion.dataset_loader_stanford import Dataset

import cv2
import supervision as L
# import spherical as S360
from sync_batchnorm import convert_model
import matplotlib.pyplot as plot
import scipy.io
from _extra.OmniFusion.model.spherical_model_iterative import spherical_fusion
#from ply import write_ply
import csv
from util import *
import shutil
import torchvision.utils as vutils 
import _extra.OmniFusion.equi_pers.equi2pers_v3
from thop import profile


#My:
from torchvision import transforms
#import torchvision.transforms as T
from PIL import Image



parser = argparse.ArgumentParser(description='PanoDepth')

parser.add_argument("--name", #required=True,
                    default=None,
                    help="Name of this run. Used for monitoring.")

parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='psmnet',
                    help='select model')
parser.add_argument('--input_dir', default='/l/users/MODE/Datasets/Stanford2D3D/',
                    help='input data directory')
parser.add_argument('--trainfile', default='/l/users/MODE/Datasets/Stanford2D3D/train_stanford2d3d.txt',
                    help='train file name')
parser.add_argument('--testfile', default='/l/users/MODE/Datasets/Stanford2D3D/test_stanford2d3d.txt',
                    help='validation file name')
parser.add_argument('--epochs', type=int, default=25, # 200
                    help='number of epochs to train')
parser.add_argument('--start_decay', type=int, default=60,
                    help='number of epoch for lr to start decay')
parser.add_argument('--start_learn', type=int, default=100,
                    help='number of iterations for stereo network to start learn')
parser.add_argument('--batch', type=int, default=2, #4,
                    help='number of batch to train')
parser.add_argument('--visualize_interval', type=int, default=20,
                    help='number of batch to train')
parser.add_argument('--baseline', type=float, default=0.24,
                    help='image pair baseline distance')
parser.add_argument('--interval', type=float, default=0.5,
                    help='second stage interval')
parser.add_argument('--nlabels', type=str, default="48,24", 
                    help='number of labels')
parser.add_argument('--checkpoint', default= None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint', default='./checkpoints',
                    help='save checkpoint path')

#parser.add_argument('--visualize_path', default='./visualize_stanford2d3d_ours_48_24',
parser.add_argument('--visualize_path', default='./output',
                    help='save checkpoint path')
parser.add_argument('--tensorboard_path', default='./logs',
                    help='tensorboard path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--real', action='store_true', default=False,
                    help='adapt to real world images in both training and validation')


## Omnifusion:
parser.add_argument('--patchsize', type=list, default=(256, 256),
                    help='patch size')
parser.add_argument('--fov', type=float, default=80,
                    help='field of view')
parser.add_argument('--iter', type=int, default=2,
                    help='number of iterations')
parser.add_argument('--nrows', type=int, default=4,
                    help='nrows, options are 4, 6')
parser.add_argument('--checkpoint_omni', default= None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint_omni', default='checkpoints',
                    help='save checkpoint path')
parser.add_argument('--save_path', default='/l/users/MODE/models/Omnifusion/',
                    help='save checkpoint path')                    
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

panodpeth_full_pretrained = True # True

#pano_coarse_pred = True # True for panodepth

panodpeth_coarse_pretrained = False # False
omnifusion_coarse_pretrained = True # True



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {'val' : self.val,
            'sum' : self.sum,
            'count' : self.count,
            'avg' : self.avg}

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']

def compute_eval_metrics2(output, gt, depth_mask):
        '''
        Computes metrics used to evaluate the model
        '''
        depth_pred = output
        gt_depth = gt

        N = depth_mask.sum()

        # Align the prediction scales via median
        median_scaling_factor = gt_depth[depth_mask>0].median() / depth_pred[depth_mask>0].median()
        depth_pred *= median_scaling_factor

        abs_rel = abs_rel_error(depth_pred, gt_depth, depth_mask)
        sq_rel = sq_rel_error(depth_pred, gt_depth, depth_mask)
        rms_sq_lin = lin_rms_sq_error(depth_pred, gt_depth, depth_mask)
        rms_sq_log = log_rms_sq_error(depth_pred, gt_depth, depth_mask)
        d1 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=1)
        d2 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=2)
        d3 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=3)
        
        abs_rel_error_meter.update(abs_rel, N)
        sq_rel_error_meter.update(sq_rel, N)
        lin_rms_sq_error_meter.update(rms_sq_lin, N)
        log_rms_sq_error_meter.update(rms_sq_log, N)
        d1_inlier_meter.update(d1, N)
        d2_inlier_meter.update(d2, N)
        d3_inlier_meter.update(d3, N)
 
def compute_eval_metrics1(output, gt, depth_mask):
        '''
        Computes metrics used to evaluate the model
        '''
        depth_pred = output
        gt_depth = gt

        N = depth_mask.sum()

        # Align the prediction scales via median
        median_scaling_factor = gt_depth[depth_mask>0].median() / depth_pred[depth_mask>0].median()
        depth_pred *= median_scaling_factor

        abs_rel = abs_rel_error(depth_pred, gt_depth, depth_mask)
        sq_rel = sq_rel_error(depth_pred, gt_depth, depth_mask)
        rms_sq_lin = lin_rms_sq_error(depth_pred, gt_depth, depth_mask)
        rms_sq_log = log_rms_sq_error(depth_pred, gt_depth, depth_mask)
        d1 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=1)
        d2 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=2)
        d3 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=3)
        
        abs_rel_error_meter_coarse.update(abs_rel, N)
        sq_rel_error_meter_coarse.update(sq_rel, N)
        lin_rms_sq_error_meter_coarse.update(rms_sq_lin, N)
        log_rms_sq_error_meter_coarse.update(rms_sq_log, N)
        d1_inlier_meter_coarse.update(d1, N)
        d2_inlier_meter_coarse.update(d2, N)
        d3_inlier_meter_coarse.update(d3, N)


abs_rel_error_meter = AverageMeter()
sq_rel_error_meter = AverageMeter()
lin_rms_sq_error_meter = AverageMeter()
log_rms_sq_error_meter = AverageMeter()
d1_inlier_meter = AverageMeter()
d2_inlier_meter = AverageMeter()
d3_inlier_meter = AverageMeter()

abs_rel_error_meter_coarse = AverageMeter()
sq_rel_error_meter_coarse = AverageMeter()
lin_rms_sq_error_meter_coarse = AverageMeter()
log_rms_sq_error_meter_coarse = AverageMeter()
d1_inlier_meter_coarse = AverageMeter()
d2_inlier_meter_coarse = AverageMeter()
d3_inlier_meter_coarse = AverageMeter()



# Save Checkpoint -------------------------------------------------------------
if not os.path.isdir(os.path.join(args.visualize_path, args.save_checkpoint)):
    os.makedirs(os.path.join(args.visualize_path, args.save_checkpoint))
    
# tensorboard Path -----------------------
writer_path = os.path.join(args.visualize_path, args.tensorboard_path)
writer = SummaryWriter(writer_path)

#-----------------------------------------

# Random Seed -----------------------------
# Panodepth ----------------------------------------------------------
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Omnifusion ----------------------------------------------------------
random.seed(args.seed) 
np.random.seed(args.seed)


# Panodepth ----------------------------------------------------------
input_dir = args.input_dir # Dataset location
train_file_list = args.trainfile # File with list of training files
val_file_list = args.testfile # File with list of validation files

#------------------------------------
result_view_dir = args.visualize_path + '/evaluation/panofusion'
if not os.path.exists(result_view_dir):
    os.makedirs(result_view_dir) 
     
#-------------------------------------------------------------------
baseline = 0.24
baseline_direction = 'vertical'

batch_size = args.batch
maxdisp = args.maxdisp
nlabels = [int(nd) for nd in args.nlabels.split(",") if nd]
visualize_interval = args.visualize_interval
interval = args.interval

lr1 = 2e-4 # original
lr2 = 5e-4 # original

# lr1 = 2e-5
# lr2 = 5e-5


# Omnifusion ----------------------------------------------------------
fov = (args.fov, args.fov)#(48, 48)
patch_size = args.patchsize
nrows = args.nrows
npatches_dict = {3:10, 4:18, 5:26, 6:46}
iters = args.iter


#-------------------------------------------data loaders
# Panodepth --------------------------------------------
train_dataset = OmniDepthDataset(
    root_path=input_dir, 
    rotate=True,
    flip=True,
    gamma=True,
    path_to_img_list=train_file_list)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size, #batch_size,
    shuffle=True,
    num_workers=8, #8,
    pin_memory=True,
    drop_last=True)

val_dataset = OmniDepthDataset(
    root_path=input_dir, 
    path_to_img_list=val_file_list)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8, #8,
    drop_last=True)


# Omnifusion ----------------------------------------------------------
# val_dataloader = torch.utils.data.DataLoader(
# 	dataset=Dataset(
# 		root_path=input_dir, 
# 		path_to_img_list=val_file_list),
# 	batch_size=2,
# 	shuffle=False,
# 	num_workers=8,
# 	drop_last=True)


# Panodepth ------------------------------------------------------
#----------------------------------------------------------
#first network, coarse depth estimation
# option 1, resnet 360 
num_gpu = torch.cuda.device_count()

first_network = ResNet360() # original
#first_network = ResNet360(batch_size//num_gpu, output_size=(256, 512), aspp=True) # from main_fullpipeline_pretrain

# #first_network = ResNet360(conv_type='standard', norm_type='batchnorm', activation='relu', aspp=True)
#weight_init.initialize_weights(first_network, init="xavier", pred_bias=float(5.0))
#first_network = RectNet()
first_network = convert_model(first_network)
# option 2, spherical unet
#view_syn_network = SphericalUnet()
first_network = nn.DataParallel(first_network)

#first_network.cuda()
#----------------------------------------------------------

# stereo matching network ----------------------------------------------
stereo_network = psmnet_spherical_up_down_inv_depth.PSMNet(nlabels, 
        [1,0.5], True, 5, cr_base_chs=[32,32,16])
stereo_network = convert_model(stereo_network)
#-----------------------------------------------------------------------------
stereo_network = nn.DataParallel(stereo_network)
#stereo_network.cuda()
#-------------------------------------------------


# Load Checkpoint -------------------------------
start_epoch = 0

if panodpeth_full_pretrained:
    #state_dict = torch.load(result_view_dir.split('/')[1] + '/checkpoints/checkpoint_latest.tar')
    #state_dict = torch.load('/l/users/MODE/models/PanoDepth/full25/full25.tar')
    #state_dict = torch.load('/l/users/MODE/models/PanoDepth/coarse10_full15/coarse10_full15.tar')

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)         
    else:
        state_dict = torch.load('/l/users/MODE/models/PanoDepth/full200/full200.tar')
    #state_dict = torch.load('/l/users/MODE/models/PanoDepth/full200/full200_decay.tar')

    #state_dict = torch.load('/l/users/MODE/models/PanoDepth/full200/panodepth_mainpipe_200.tar')
    #state_dict = torch.load('/l/users/MODE/models/PanoDepth/coarse10_full200/coarse10_full200.tar')
    #state_dict = torch.load('/l/users/MODE/models/PanoDepth/coarse10_full200/coarse10_full200_decay.tar')

    first_network.load_state_dict(state_dict['state_dict1'])
    stereo_network.load_state_dict(state_dict['state_dict2'])
    #start_epoch = state_dict['epoch']

if panodpeth_coarse_pretrained:
    state_dict_first = torch.load('/l/users/MODE/models/PanoDepth/coarse10/coarse10.tar')
    first_network.load_state_dict(state_dict_first)

# Omnifusion ----------------------------------------------------------
if omnifusion_coarse_pretrained:
    first_network = spherical_fusion()
    first_network = convert_model(first_network)
    # parallel on multi gpu
    first_network = nn.DataParallel(first_network)
    ckpt = torch.load(args.save_path + 'Stanford_2iter.pth')

    first_network.load_state_dict(ckpt)
    #first_network.cuda()

    print('## patch size:', patch_size) 
    print('## fov:', args.fov)
    print('## Number of first model (OmniFusion) parameters: {}'.format(sum([p.data.nelement() for p in first_network.parameters() if p.requires_grad is True])))
#-------------------------------------------------

first_network.cuda()
stereo_network.cuda()


print('## Batch size: {}'.format(batch_size))  
print('## learning rate 1: {}, learning rate 2: {}'.format(lr1, lr2))  
print('## Number of first model parameters: {}'.format(sum([p.data.nelement() for p in first_network.parameters()])))
print('## Number of stereo matching model parameters: {}'.format(sum([p.data.nelement() for p in stereo_network.parameters()])))
#--------------------------------------------------

# Optimizer ----------
#optimizer = optim.Adam(list(model.parameters())+list(view_syn_network.parameters()), 
#        lr=0.001, betas=(0.9, 0.999))
optimizer1 = optim.Adam(list(first_network.parameters()), 
        lr=lr1, betas=(0.9, 0.999))
optimizer2 = optim.Adam(list(stereo_network.parameters()), 
        lr=lr2, betas=(0.9, 0.999))

# scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, [20, 40, 60], 0.5)
# scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, [20, 40, 60], 0.5)

scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, [7, 15, 22], 0.5)
scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, [7, 15, 22], 0.5)

# scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, [30, 60, 90, 120], 0.5)
# scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, [30, 60, 90, 120], 0.5)

# scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, [30, 60, 90, 120, 150, 180], 0.5)
# scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, [30, 60, 90, 120, 150, 180], 0.5)

# scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, [20, 40, 60, 80, 100, 120, 140, 160, 180], 0.5)
# scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, [20, 40, 60, 80, 100, 120, 140, 160, 180], 0.5)
#---------------------

    
# Train Function -------------------
def train(target, render, depth_down, mask, batch_idx):
    #stereo_network.train()
    #mask = mask>0

    # mask value
    #mask = (disp_true < args.maxdisp) & (disp_true > 0)
    mask = mask>0
    mask.detach_()

    # Loss -------------------------------------------- 
    #outputs = stereo_network(render[0], target, baseline=baseline, direction=baseline_direction)
    outputs = stereo_network(render, target)
    #sgrid = S360.grid.create_spherical_grid(512).cuda()
    #disparity = S360.derivatives.dtheta_vertical(sgrid, depth_down.unsqueeze(1), baseline).squeeze(1)
    stereo_loss = stereo_psmnet_loss(outputs, depth_down, mask, dlossw=[0.5, 2.0])
    #loss = disp_loss
    #--------------------------------------------------
    output3 = outputs["stage2"]["pred"]
    #output3 = S360.derivatives.disparity_to_depth_vertical(sgrid, output3.unsqueeze(1), baseline).squeeze(1)
    gt = target[:,:3,:,:].detach().cpu().numpy()
    render_np = render.detach().cpu().numpy()
    depth = depth_down.detach().cpu().numpy()
    depth_prediction = output3.detach().cpu().numpy()


    #depth_prediction[depth_prediction>8] = 0 #!!!!
    depth_prediction[depth_prediction < 0.1] = 0.1
    depth_prediction[depth_prediction > 8] = 8
    
    if batch_idx % visualize_interval == 0 and batch_idx > 0:
            gt_img = gt[0, :, :, :].transpose(1,2,0)
            depth_img = depth[0, :, :]
            #depth_down_img = cv2.normalize(depth_down_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            depth_pred_img = depth_prediction[0, :, :]
            #depth_pred_img = cv2.normalize(depth_pred_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.imwrite('{}/train_gt_{}.png'.format(result_view_dir, batch_idx), gt_img*255)
            render_img = render_np[0, :, :, :].transpose(1,2,0)
            cv2.imwrite('{}/train_render_{}.png'.format(result_view_dir, batch_idx), render_img*255)
            plot.imsave('{}/train_depth_gt_{}.png'.format(result_view_dir, batch_idx), depth_img, cmap="jet")
            plot.imsave('{}/train_depth_pred_{}.png'.format(result_view_dir, batch_idx), depth_pred_img, cmap="jet")

    return output3, stereo_loss

# Valid Function -----------------------
def val(target, render, depth, mask, batch_idx):
    stereo_network.eval()

    sgrid = S360.grid.create_spherical_grid(512).cuda()

    with torch.no_grad():
        #outputs = stereo_network(render[0], target, baseline=baseline, direction=baseline_direction)
        outputs =stereo_network(render, target)
        output3 = outputs["stage2"]["pred"]
        #output3 = S360.derivatives.disparity_to_depth_vertical(sgrid, output3.unsqueeze(1), baseline).squeeze(1)

    gt = target[:,:3,:,:].detach().cpu().numpy()
    render_np = render.detach().cpu().numpy()

    depth = depth.detach().cpu().numpy()
    sgrid = S360.grid.create_spherical_grid(512).cuda()
    depth_prediction = output3.detach().cpu().numpy()

    if batch_idx % 20 == 0 and batch_idx > 0:
            gt_img = gt[0, :, :, :].transpose(1,2,0)

            depth_img = depth[0, :, :]

            # print("Before:")
            # print(max(depth_img).any())
            # print(min(depth_img).any())

            # depth_img[depth_img < 0.2] = 0.2
            # depth_img[depth_img > 8] = 8

            # print("After:")
            # print((max(depth_img)).all())
            # print((min(depth_img)).any())

            #depth_down_img = cv2.normalize(depth_down_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            depth_pred_img = depth_prediction[0, :, :]

            #depth_prediction[depth_prediction>8] = 0
            depth_pred_img[depth_pred_img < 0.1] = 0.1
            depth_pred_img[depth_pred_img > 8] = 8

            # depth_pred_img = (depth_pred_img / 8 * 65535).astype(np.uint16) # from main func
            
            #depth_pred_img = cv2.normalize(depth_pred_img,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.imwrite('{}/test_gt_{}.png'.format(result_view_dir, batch_idx), gt_img*255)
            render_img = render_np[0, :, :, :].transpose(1,2,0)
            cv2.imwrite('{}/test_render_{}.png'.format(result_view_dir, batch_idx), render_img*255)
            plot.imsave('{}/test_depth_gt_{}.png'.format(result_view_dir, batch_idx), depth_img, cmap="jet")
            plot.imsave('{}/test_depth_pred_{}.png'.format(result_view_dir, batch_idx), depth_pred_img, cmap="jet")

    return output3



 
# Main Function ---------------------------------------------------------------------------------------------
def main():
    global_step = 0
    global_val = 0

    csv_filename = os.path.join(result_view_dir, 'logs/result_log.csv')
    fields = ['epoch', 'Abs Rel', 'Sq Rel', 'Lin RMSE', 'log RMSE', 'D1', 'D2', 'D3']
    csvfile = open(csv_filename, 'w', newline='')
    with csvfile:
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(fields) 

    if not omnifusion_coarse_pretrained:
        print("Panodepth coarse")
    else:
        print("Omnifusion coarse")

    # Start Training ---------------------------------------------------------
    start_full_time = time.time()
    for epoch in tqdm(range(start_epoch+1, args.epochs+1), desc='Epoch'):
        print('---------------Train Epoch', epoch, '----------------')
        total_train_loss = 0
        first_depth_estimation_loss = 0
        stereo_matching_loss = 0
        #-------------------------------

        #first_network.eval() # original
        first_network.train() # our thought

        stereo_network.train()
        # Train --------------------------------------------------------------------------------------------------
        for batch_idx, (rgb, depth, depth_mask) in tqdm(enumerate(train_dataloader)):
            #get ground truth up-down disparity
            sgrid = S360.grid.create_spherical_grid(512).cuda()
            uvgrid = S360.grid.create_image_grid(512, 256).cuda()
            rgb, depth, depth_mask = rgb.cuda(), depth.cuda(), (depth_mask>0).cuda()

            #first depth estimation
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            if not omnifusion_coarse_pretrained:
                coarse_depth_pred = first_network(rgb[:,:3,:,:])
            
            else:
                # Omnifusion ----------------------------------------------------------
                #print("Omnifusion coarse")
                
                batching = True # True
                batching_torch = False # True
                batching_torch_full = False # True

                resize_input_img = True

                if resize_input_img:

                    if batching:
                        if batching_torch:
                            if batching_torch_full:

                                # transform_img = transforms.Compose([
                                #     #transforms.ToTensor(),
                                #     transforms.ToPILImage(),
                                #     transforms.Resize((512, 1024), transforms.InterpolationMode.BILINEAR),
                                #     transforms.ToTensor(),
                                # ])
                                
                                #img_final = transform_img(rgb)

                                #print("rgb1.5", rgb.size())

                                img_final = transforms.Resize((512, 1024), transforms.InterpolationMode.BILINEAR)(rgb)

                                #print("rgb1.5", img_final.size())

                            else:
                                for i in range(rgb.size()[0]):
                                    img_tmp = rgb[i,:3,:,:]

                                    transform_img = transforms.Compose([
                                        #transforms.ToTensor(),
                                        transforms.ToPILImage(),
                                        transforms.Resize((512, 1024), transforms.InterpolationMode.BILINEAR),
                                        transforms.ToTensor(),
                                    ])

                                    #print("before", img_tmp.size())

                                    #img_tmp = transform_img(img_tmp)
                                    img_tmp = transforms.Resize((512, 1024), transforms.InterpolationMode.BILINEAR)(img_tmp)

                                    #print("after", img_tmp.size())

                                    img_tmp = img_tmp.unsqueeze(0)

                                    if i < 1:
                                        img_final = img_tmp
                                    else:
                                        #img_final = np.concatenate(res,axis=1)
                                        img_final = torch.cat((img_final, img_tmp), 0)
                        
                                #print("rgb2", img_final.size())
                            
                            img_final = img_final.float().cuda()

                        else:
                            for i in range(rgb.size()[0]):
                                img_tmp = rgb[i,:3,:,:]

                                img_tmp = img_tmp.detach().cpu().numpy()

                                img_tmp = img_tmp.transpose(1, 2, 0)

                                #print(img_tmp.shape)
                                img_tmp = cv2.resize(img_tmp, (1024, 512), interpolation = cv2.INTER_AREA)
                                #print(img_tmp.shape)
                                img_tmp = img_tmp.transpose(2, 0, 1)
                                #print(img_tmp.shape)
                                img_tmp = np.expand_dims(img_tmp, 0)
                                #print(img_tmp.shape)

                                #img_tmp = np.repeat(img_tmp, 2, axis=0)


                                img_resized = torch.from_numpy(img_tmp).float().cuda()

                                img_final_tmp = img_resized
                                if i < 1:
                                    img_final = img_final_tmp
                                else:
                                    #img_final = np.concatenate(res,axis=1)
                                    img_final = torch.cat((img_final, img_final_tmp), 0)
                            #img_final = img_final.float().cuda()
                            img_final = img_final.cuda()


                    else:

                        #img_tmp = rgb[0,:3,:,:]

                        #print(rgb.size())
                        # print(rgb)
                        # print(rgb.max())

                        img_tmp = rgb[0,:3,:,:].detach().cpu().numpy()
                        #print(img_tmp.shape)

                        # print(img_tmp)
                        # print(img_tmp.max())

                        img_tmp = img_tmp.transpose(1, 2, 0)

                        #print(img_tmp.shape)
                        img_tmp = cv2.resize(img_tmp, (1024, 512), interpolation = cv2.INTER_AREA)
                        #print(img_tmp.shape)
                        img_tmp = img_tmp.transpose(2, 0, 1)
                        #print(img_tmp.shape)
                        img_tmp = np.expand_dims(img_tmp, 0)
                        #print(img_tmp.shape)

                        #img_tmp = np.repeat(img_tmp, 2, axis=0)

                        img_final = torch.from_numpy(img_tmp).float().cuda()

                        # print('==omni===')
                        # print(img_resized.size())


                #first_network.eval()
                # #for batch_idx, (rgb, depth, mask) in tqdm(enumerate(val_dataloader)):
                #     #rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()
                # bs, _, h, w = img_omni.shape
                # print(h, w)
                
                # with torch.no_grad():
                #img_final = img_final.cuda()

                if resize_input_img:
                    equi_outputs_list = first_network(img_final, iter=iters)
                else:
                    equi_outputs_list = first_network(rgb, iter=iters)

                #equi_outputs = equi_outputs_list[-1]
                coarse_depth_pred = equi_outputs_list[-1]

                #print('lol', coarse_depth_pred.shape)


                if batching:
                    if batching_torch:
                        if batching_torch_full:
                            #print("coarse1.5", coarse_depth_pred.size())

                            #coarse_depth_pred = transforms.Resize((256, 512), transforms.InterpolationMode.BILINEAR)(coarse_depth_pred)
                            coarse_depth_pred = transforms.Resize((256, 512), transforms.InterpolationMode.NEAREST)(coarse_depth_pred)
                            # TODO try another interpolation!!!
                            #print("coarse1.5", coarse_depth_pred.size())

                        else:
                            for i in range(coarse_depth_pred.size()[0]):
                                coarse_tmp = coarse_depth_pred[i,:3,:,:]

                                transform_coarse = transforms.Compose([
                                    #transforms.ToTensor(),
                                    transforms.ToPILImage(),
                                    transforms.Resize((256, 512), transforms.InterpolationMode.NEAREST), #BILINEAR),
                                    #transforms.Resize((512, 256), transforms.InterpolationMode.NEAREST), #BILINEAR),
                                    transforms.ToTensor(),
                                ])

                                #print("rgb1.5", rgb.size())
                                #print("before", coarse_tmp.size())

                                #coarse_tmp = transform_coarse(coarse_tmp)
                                coarse_tmp = transforms.Resize((256, 512), transforms.InterpolationMode.NEAREST)(coarse_tmp)

                                #print("rgb1.5", img_tmp.size())
                                #print("after 0.5", coarse_tmp.size())

                                coarse_tmp = coarse_tmp.unsqueeze(0)
                                #print("after", coarse_tmp.size())

                                if i < 1:
                                    coarse_final = coarse_tmp
                                else:
                                    #img_final = np.concatenate(res,axis=1)
                                    coarse_final = torch.cat((coarse_final, coarse_tmp), 0)

                            coarse_depth_pred_final = coarse_final
                            # print("final", coarse_depth_pred_final.size())

                            #print("rgb2", img_final.size())
                            coarse_depth_pred = coarse_depth_pred_final.cuda()

                    else:
                        for i in range(coarse_depth_pred.size()[0]):
                            coarse_tmp = coarse_depth_pred[i,:,:,:]
                            #print('lol_temp', coarse_depth_pred.shape)

                            depth_prediction = coarse_tmp.detach().cpu().numpy()

                            # #depth_prediction[depth_prediction > 8] = 0
                            # depth_prediction[depth_prediction < 0.2] = 0.2
                            # depth_prediction[depth_prediction > 8] = 8

                            depth_pred_img = depth_prediction[0, :, :]

                            #print(depth_pred_img.shape)
                            depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_AREA) # better acc
                            #depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_NEAREST) # supposed to be used for masks
                            #depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_NEAREST_EXACT) # more detailed

                            #print(depth_pred_img_small.shape)


                            #coarse_depth_pred[0, 0, :, :] = depth_pred_img_small
                            depth_pred_img_small = np.expand_dims(depth_pred_img_small, 0)
                            coarse_depth_pred_tmp = torch.from_numpy(depth_pred_img_small).float().cuda()
                            coarse_depth_pred_tmp = coarse_depth_pred_tmp.unsqueeze(0)


                            coarse_final_tmp = coarse_depth_pred_tmp
                            if i < 1:
                                coarse_final = coarse_final_tmp
                            else:
                                #img_final = np.concatenate(res,axis=1)
                                coarse_final = torch.cat((coarse_final, coarse_final_tmp), 0)

                        coarse_depth_pred_final = coarse_final
                        #print('lol2', coarse_depth_pred_final.shape)

                        #coarse_depth_pred = coarse_depth_pred_final
                        coarse_depth_pred = coarse_depth_pred_final.cuda()

                else:
                    depth_prediction = coarse_depth_pred.detach().cpu().numpy()

                    # #depth_prediction[depth_prediction > 8] = 0
                    # depth_prediction[depth_prediction < 0.2] = 0.2
                    # depth_prediction[depth_prediction > 8] = 8
                    
                    depth_pred_img = depth_prediction[0, 0, :, :]

                    #print(depth_pred_img.shape)
                    #depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_AREA)
                    depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_NEAREST)
                    #print(depth_pred_img_small.shape)


                    #coarse_depth_pred[0, 0, :, :] = depth_pred_img_small
                    depth_pred_img_small = np.expand_dims(depth_pred_img_small, 0)
                    coarse_depth_pred = torch.from_numpy(depth_pred_img_small).float().cuda()
                    coarse_depth_pred = coarse_depth_pred.unsqueeze(0)



                '''
                depth_prediction = coarse_depth_pred.detach().cpu().numpy()
                depth_prediction[depth_prediction > 8] = 0
                depth_pred_img_1 = depth_prediction[0, 0, :, :]
                depth_pred_img_2 = depth_prediction[1, 0, :, :]

                print(depth_pred_img_1.shape)
                #depth_pred_img_small_1 = cv2.resize(depth_pred_img_1, (512, 256), interpolation = cv2.INTER_AREA)
                depth_pred_img_small_1 = cv2.resize(depth_pred_img_1, (512, 256), interpolation = cv2.INTER_NEAREST)

                
                print(depth_pred_img_small_1.shape)

                print(depth_pred_img_2.shape)
                #depth_pred_img_small_2 = cv2.resize(depth_pred_img_2, (512, 256), interpolation = cv2.INTER_AREA)
                depth_pred_img_small_2 = cv2.resize(depth_pred_img_1, (512, 256), interpolation = cv2.INTER_NEAREST)

                print(depth_pred_img_small_2.shape)

                #coarse_depth_pred[0, 0, :, :] = depth_pred_img_small
                depth_pred_img_small_1 = np.expand_dims(depth_pred_img_small_1, 0)
                depth_pred_img_small_2 = np.expand_dims(depth_pred_img_small_2, 0)                
                depth_pred_img_small_1 = torch.from_numpy(depth_pred_img_small_1).float().cuda()
                depth_pred_img_small_2 = torch.from_numpy(depth_pred_img_small_2).float().cuda()


                coarse_depth_pred = torch.stack((depth_pred_img_small_1,depth_pred_img_small_2))
                '''

                '''
                #coarse_depth_pred = transforms.ToTensor()(coarse_depth_pred)
                coarse_depth_pred =  transforms.ToPILImage()(coarse_depth_pred)
        
                #for 
                coarse_depth_pred = transforms.Resize((512, 256), interpolation=Image.NEAREST)(coarse_depth_pred), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                coarse_depth_pred = transforms.ToTensor()(coarse_depth_pred)
                '''
                #print(coarse_depth_pred.size())

                # error = torch.abs(depth - equi_outputs) * mask
                # error[error < 0.1] = 0
                
                
                '''
                #compute_eval_metrics(equi_outputs, depth, mask)
                rgb_img = img.detach().cpu().numpy()
                depth_prediction = equi_outputs.detach().cpu().numpy()
                equi_gt = depth.detach().cpu().numpy()
                #error_img = error.detach().cpu().numpy()
                depth_prediction[depth_prediction > 8] = 0
                
                # save raw 3D point cloud reconstruction as ply file
                coords = np.stack(np.meshgrid(range(w), range(h)), -1)
                coords = np.reshape(coords, [-1, 2])
                coords += 1
                uv = coords2uv(coords, w, h)          
                xyz = uv2xyz(uv) 
                xyz = torch.from_numpy(xyz).to(img.device)
                xyz = xyz.unsqueeze(0).repeat(bs, 1, 1)
                gtxyz = xyz * depth.reshape(bs, w*h, 1)    
                predxyz = xyz * equi_outputs.reshape(bs, w*h, 1)
                gtxyz = gtxyz.detach().cpu().numpy()
                predxyz = predxyz.detach().cpu().numpy()
                #error = error.detach().cpu().numpy()

                #if batch_idx % 20 == 0:
                rgb_img = rgb_img[0, :, :, :].transpose(1, 2, 0)
                depth_pred_img = depth_prediction[0, 0, :, :]
                print(depth_pred_img)
                print(depth_pred_img.max())

                print(depth_pred_img.shape)
                depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_AREA)
                print(depth_pred_img_small.shape)


                depth_gt_img = equi_gt[0, 0, :, :]
                error_img = error_img[0, 0, :, :] 
                gtxyz_np = predxyz[0, ...]
                predxyz_np = predxyz[0, ...]
                batch_idx = 0
                cv2.imwrite('{}/test_equi_rgb_{}.png'.format(result_view_dir, batch_idx),
                            rgb_img*255)
                plot.imsave('{}/test_equi_pred_{}.png'.format(result_view_dir, batch_idx),
                            depth_pred_img, cmap="jet")
                plot.imsave('{}/test_equi_pred_small_{}.png'.format(result_view_dir, batch_idx),
                            depth_pred_img_small, cmap="jet")                
                plot.imsave('{}/test_equi_gt_{}.png'.format(result_view_dir, batch_idx),
                            depth_gt_img, cmap="jet")
                plot.imsave('{}/test_error_{}.png'.format(result_view_dir, batch_idx),
                            error_img, cmap="jet")
                rgb_img = np.reshape(rgb_img*255, (-1, 3)).astype(np.uint8)
                #write_ply('{}/test_gt_{}'.format(result_view_dir, batch_idx), [gtxyz_np, rgb_img], ['x', 'y', 'z', 'blue', 'green', 'red'])
                #write_ply('{}/test_pred_{}'.format(result_view_dir, batch_idx), [predxyz_np, rgb_img], ['x', 'y', 'z', 'blue', 'green', 'red'])
                '''



            attention_weights = S360.weights.theta_confidence(sgrid)
            # berhu depth loss with coordinates attention weight 
            depth_loss = L.direct.calculate_berhu_loss(coarse_depth_pred, depth,                
               mask=depth_mask, weights=attention_weights)
            left_xyz = S360.cartesian.coords_3d(sgrid, coarse_depth_pred)
            dI_dxyz = S360.derivatives.dV_dxyz(left_xyz)               
            guidance_duv = S360.derivatives.dI_duv(rgb[:,:3,:,:])
            depth_smoothness_loss = L.smoothness.guided_smoothness_loss(
                dI_dxyz, guidance_duv, depth_mask, (1.0 - attention_weights)
                * depth_mask.type(attention_weights.dtype)
            )

            coarse_loss = depth_loss #+ depth_smoothness_loss * 0.1
            
            depth_np = coarse_depth_pred.detach().cpu().numpy()
            #print("Min depth:", depth_np.min(), "Max depth:", depth_np.max())

            if batch_idx % 20 == 0 and batch_idx > 0:
                    
                depth_coarse_img = depth_np[0, 0, :, :]

                #depth_coarse_img[depth_coarse_img>8] = 0
                depth_coarse_img[depth_coarse_img < 0.1] = 0.1
                depth_coarse_img[depth_coarse_img > 8] = 8

                depth_coarse_img = (depth_coarse_img / 8 * 65535).astype(np.uint16)
                plot.imsave('{}/train_coarse_depth_{}.png'.format(result_view_dir, batch_idx), depth_coarse_img, cmap='jet')
            
            render = dibr_vertical(coarse_depth_pred.clamp(0.1, 8.0), rgb, uvgrid, sgrid, baseline=baseline)
            output, stereo_loss = train(rgb, render, depth.squeeze(1), depth_mask.squeeze(1), batch_idx)
            #need to balance the first network loss and second stereo matching network loss
            alpha, beta = 0.02, 1
            
            loss = stereo_loss*0.02 + coarse_loss 
            #print(stereo_loss.item(), coarse_loss.item(), vgg_loss.item())
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            #coarse_loss = torch.tensor(0)
            total_train_loss += loss.item()
            first_depth_estimation_loss += coarse_loss.item() 
            stereo_matching_loss += stereo_loss.item() 
            global_step += 1
            if batch_idx % 20 == 0 and batch_idx > 0:
                print('[Epoch %d--Iter %d]total loss %.4f, coarse loss %.4f, stereo loss %.4f' % 
                (epoch, batch_idx, total_train_loss/(batch_idx+1), first_depth_estimation_loss/(batch_idx+1), stereo_matching_loss/(batch_idx+1)))
            writer.add_scalar('first network depth loss', coarse_loss, global_step)
            writer.add_scalar('stereo matching loss', stereo_loss, global_step)
            writer.add_scalar('total loss', loss, global_step) # tensorboardX for iter
        writer.add_scalar('train loss epoch',total_train_loss/len(train_dataloader),epoch) # tensorboardX for epoch
        writer.add_scalar('stereo loss epoch',stereo_matching_loss/len(train_dataloader),epoch)
        writer.add_scalar('coarse loss epoch',first_depth_estimation_loss/len(train_dataloader),epoch)
        #---------------------------------------------------------------------------------------------------------
        scheduler1.step()
        scheduler2.step()

        if epoch % 10 == 0:
            epochfilename = args.visualize_path + '/checkpoints/checkpoint_' + str(args.name) + str(epoch) + '.tar'  
            torch.save({
                    'state_dict1': first_network.state_dict(),
                    'state_dict2': stereo_network.state_dict(),
                    }, epochfilename)
        latestfilename = args.visualize_path + '/checkpoints/checkpoint_latest_' + str(args.name) + '.tar'
        torch.save({
                'state_dict1': first_network.state_dict(),
                'state_dict2': stereo_network.state_dict(),
                }, latestfilename)
        #-----------------------------------------------------------------------------

        # Valid ----------------------------------------------------------------------------------------------------
        total_val_loss = 0
        total_val_crop_rmse = 0
        print('-------------Validate Epoch', epoch, '-----------')

        first_network.eval()
        stereo_network.eval()
        for batch_idx, (rgb, depth, depth_mask) in tqdm(enumerate(val_dataloader)):
            sgrid = S360.grid.create_spherical_grid(512).cuda()
            uvgrid = S360.grid.create_image_grid(512, 256).cuda()
            rgb, depth, depth_mask = rgb.cuda(), depth.cuda(), (depth_mask>0).cuda()


            if not omnifusion_coarse_pretrained:
                with torch.no_grad():
                    coarse_depth_pred = first_network(rgb[:,:3,:,:])
            
    
            else:
                # Omnifusion ----------------------------------------------------------
                #print("Omnifusion coarse")
                
                batching = True # True
                batching_torch = False # True
                batching_torch_full = False # True

                resize_input_img = True

                if resize_input_img:

                    if batching:
                        if batching_torch:
                            if batching_torch_full:

                                # transform_img = transforms.Compose([
                                #     #transforms.ToTensor(),
                                #     transforms.ToPILImage(),
                                #     transforms.Resize((512, 1024), transforms.InterpolationMode.BILINEAR),
                                #     transforms.ToTensor(),
                                # ])
                                
                                #img_final = transform_img(rgb)

                                #print("rgb1.5", rgb.size())

                                img_final = transforms.Resize((512, 1024), transforms.InterpolationMode.BILINEAR)(rgb)

                                #print("rgb1.5", img_final.size())

                            else:
                                for i in range(rgb.size()[0]):
                                    img_tmp = rgb[i,:3,:,:]

                                    transform_img = transforms.Compose([
                                        #transforms.ToTensor(),
                                        transforms.ToPILImage(),
                                        transforms.Resize((512, 1024), transforms.InterpolationMode.BILINEAR),
                                        transforms.ToTensor(),
                                    ])

                                    #print("before", img_tmp.size())

                                    #img_tmp = transform_img(img_tmp)
                                    img_tmp = transforms.Resize((512, 1024), transforms.InterpolationMode.BILINEAR)(img_tmp)

                                    #print("after", img_tmp.size())

                                    img_tmp = img_tmp.unsqueeze(0)

                                    if i < 1:
                                        img_final = img_tmp
                                    else:
                                        #img_final = np.concatenate(res,axis=1)
                                        img_final = torch.cat((img_final, img_tmp), 0)
                        
                                #print("rgb2", img_final.size())
                            
                            img_final = img_final.float().cuda()


                        else:
                            for i in range(rgb.size()[0]):
                                img_tmp = rgb[i,:3,:,:]

                                img_tmp = img_tmp.detach().cpu().numpy()

                                img_tmp = img_tmp.transpose(1, 2, 0)

                                #print(img_tmp.shape)
                                img_tmp = cv2.resize(img_tmp, (1024, 512), interpolation = cv2.INTER_AREA)
                                #print(img_tmp.shape)
                                img_tmp = img_tmp.transpose(2, 0, 1)
                                #print(img_tmp.shape)
                                img_tmp = np.expand_dims(img_tmp, 0)
                                #print(img_tmp.shape)

                                #img_tmp = np.repeat(img_tmp, 2, axis=0)


                                img_resized = torch.from_numpy(img_tmp).float().cuda()

                                img_final_tmp = img_resized
                                if i < 1:
                                    img_final = img_final_tmp
                                else:
                                    #img_final = np.concatenate(res,axis=1)
                                    img_final = torch.cat((img_final, img_final_tmp), 0)
                            #img_final = img_final.float().cuda()
                            img_final = img_final.cuda()


                    else:

                        #img_tmp = rgb[0,:3,:,:]

                        #print(rgb.size())
                        # print(rgb)
                        # print(rgb.max())

                        img_tmp = rgb[0,:3,:,:].detach().cpu().numpy()
                        #print(img_tmp.shape)

                        # print(img_tmp)
                        # print(img_tmp.max())

                        img_tmp = img_tmp.transpose(1, 2, 0)

                        #print(img_tmp.shape)
                        img_tmp = cv2.resize(img_tmp, (1024, 512), interpolation = cv2.INTER_AREA)
                        #print(img_tmp.shape)
                        img_tmp = img_tmp.transpose(2, 0, 1)
                        #print(img_tmp.shape)
                        img_tmp = np.expand_dims(img_tmp, 0)
                        #print(img_tmp.shape)

                        #img_tmp = np.repeat(img_tmp, 2, axis=0)

                        img_final = torch.from_numpy(img_tmp).float().cuda()

                        # print('==omni===')
                        # print(img_resized.size())


                #first_network.eval()
                # #for batch_idx, (rgb, depth, mask) in tqdm(enumerate(val_dataloader)):
                #     #rgb, depth, mask = rgb.cuda(), depth.cuda(), mask.cuda()
                # bs, _, h, w = img_omni.shape
                # print(h, w)
                
                with torch.no_grad():
                    #img_final = img_final.cuda()

                    if resize_input_img:
                        equi_outputs_list = first_network(img_final, iter=iters)
                    else:
                        equi_outputs_list = first_network(rgb, iter=iters)

                    #equi_outputs = equi_outputs_list[-1]
                    coarse_depth_pred = equi_outputs_list[-1]

                    #print('lol', coarse_depth_pred.shape)


                    if batching:
                        if batching_torch:
                            if batching_torch_full:
                                #print("coarse1.5", coarse_depth_pred.size())

                                #coarse_depth_pred = transforms.Resize((256, 512), transforms.InterpolationMode.BILINEAR)(coarse_depth_pred)
                                coarse_depth_pred = transforms.Resize((256, 512), transforms.InterpolationMode.NEAREST)(coarse_depth_pred)
                                # TODO try another interpolation!!!
                                #print("coarse1.5", coarse_depth_pred.size())

                            else:
                                for i in range(coarse_depth_pred.size()[0]):
                                    coarse_tmp = coarse_depth_pred[i,:3,:,:]

                                    transform_coarse = transforms.Compose([
                                        #transforms.ToTensor(),
                                        transforms.ToPILImage(),
                                        transforms.Resize((256, 512), transforms.InterpolationMode.NEAREST), #BILINEAR),
                                        #transforms.Resize((512, 256), transforms.InterpolationMode.NEAREST), #BILINEAR),
                                        transforms.ToTensor(),
                                    ])

                                    #print("rgb1.5", rgb.size())
                                    #print("before", coarse_tmp.size())

                                    #coarse_tmp = transform_coarse(coarse_tmp)
                                    coarse_tmp = transforms.Resize((256, 512), transforms.InterpolationMode.NEAREST)(coarse_tmp)

                                    #print("rgb1.5", img_tmp.size())
                                    #print("after 0.5", coarse_tmp.size())

                                    coarse_tmp = coarse_tmp.unsqueeze(0)
                                    #print("after", coarse_tmp.size())

                                    if i < 1:
                                        coarse_final = coarse_tmp
                                    else:
                                        #img_final = np.concatenate(res,axis=1)
                                        coarse_final = torch.cat((coarse_final, coarse_tmp), 0)

                                coarse_depth_pred_final = coarse_final
                                # print("final", coarse_depth_pred_final.size())

                                #print("rgb2", img_final.size())
                                coarse_depth_pred = coarse_depth_pred_final.cuda()

                        else:
                            for i in range(coarse_depth_pred.size()[0]):
                                coarse_tmp = coarse_depth_pred[i,:,:,:]
                                #print('lol_temp', coarse_depth_pred.shape)

                                depth_prediction = coarse_tmp.detach().cpu().numpy()

                                # #depth_prediction[depth_prediction > 8] = 0
                                # depth_prediction[depth_prediction < 0.2] = 0.2
                                # depth_prediction[depth_prediction > 8] = 8

                                depth_pred_img = depth_prediction[0, :, :]

                                #print(depth_pred_img.shape)
                                depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_AREA) # better acc
                                #depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_NEAREST) # supposed to be used for masks
                                #depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_NEAREST_EXACT) # more detailed

                                #print(depth_pred_img_small.shape)


                                #coarse_depth_pred[0, 0, :, :] = depth_pred_img_small
                                depth_pred_img_small = np.expand_dims(depth_pred_img_small, 0)
                                coarse_depth_pred_tmp = torch.from_numpy(depth_pred_img_small).float().cuda()
                                coarse_depth_pred_tmp = coarse_depth_pred_tmp.unsqueeze(0)


                                coarse_final_tmp = coarse_depth_pred_tmp
                                if i < 1:
                                    coarse_final = coarse_final_tmp
                                else:
                                    #img_final = np.concatenate(res,axis=1)
                                    coarse_final = torch.cat((coarse_final, coarse_final_tmp), 0)

                            coarse_depth_pred_final = coarse_final
                            #print('lol2', coarse_depth_pred_final.shape)

                            #coarse_depth_pred = coarse_depth_pred_final
                            coarse_depth_pred = coarse_depth_pred_final.cuda()

                    else:
                        depth_prediction = coarse_depth_pred.detach().cpu().numpy()

                        # #depth_prediction[depth_prediction > 8] = 0
                        # depth_prediction[depth_prediction < 0.2] = 0.2
                        # depth_prediction[depth_prediction > 8] = 8
                        
                        depth_pred_img = depth_prediction[0, 0, :, :]

                        #print(depth_pred_img.shape)
                        #depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_AREA)
                        depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_NEAREST)
                        #print(depth_pred_img_small.shape)


                        #coarse_depth_pred[0, 0, :, :] = depth_pred_img_small
                        depth_pred_img_small = np.expand_dims(depth_pred_img_small, 0)
                        coarse_depth_pred = torch.from_numpy(depth_pred_img_small).float().cuda()
                        coarse_depth_pred = coarse_depth_pred.unsqueeze(0)




                    '''
                    depth_prediction = coarse_depth_pred.detach().cpu().numpy()
                    depth_prediction[depth_prediction > 8] = 0
                    depth_pred_img_1 = depth_prediction[0, 0, :, :]
                    depth_pred_img_2 = depth_prediction[1, 0, :, :]

                    print(depth_pred_img_1.shape)
                    #depth_pred_img_small_1 = cv2.resize(depth_pred_img_1, (512, 256), interpolation = cv2.INTER_AREA)
                    depth_pred_img_small_1 = cv2.resize(depth_pred_img_1, (512, 256), interpolation = cv2.INTER_NEAREST)

                    
                    print(depth_pred_img_small_1.shape)

                    print(depth_pred_img_2.shape)
                    #depth_pred_img_small_2 = cv2.resize(depth_pred_img_2, (512, 256), interpolation = cv2.INTER_AREA)
                    depth_pred_img_small_2 = cv2.resize(depth_pred_img_1, (512, 256), interpolation = cv2.INTER_NEAREST)

                    print(depth_pred_img_small_2.shape)

                    #coarse_depth_pred[0, 0, :, :] = depth_pred_img_small
                    depth_pred_img_small_1 = np.expand_dims(depth_pred_img_small_1, 0)
                    depth_pred_img_small_2 = np.expand_dims(depth_pred_img_small_2, 0)                
                    depth_pred_img_small_1 = torch.from_numpy(depth_pred_img_small_1).float().cuda()
                    depth_pred_img_small_2 = torch.from_numpy(depth_pred_img_small_2).float().cuda()


                    coarse_depth_pred = torch.stack((depth_pred_img_small_1,depth_pred_img_small_2))
                    '''

                    '''
                    #coarse_depth_pred = transforms.ToTensor()(coarse_depth_pred)
                    coarse_depth_pred =  transforms.ToPILImage()(coarse_depth_pred)
            
                    #for 
                    coarse_depth_pred = transforms.Resize((512, 256), interpolation=Image.NEAREST)(coarse_depth_pred), #Image.BILINEAR), # interpolation=T.InterpolationMode.NEAREST
                    coarse_depth_pred = transforms.ToTensor()(coarse_depth_pred)
                    '''
                    #print(coarse_depth_pred.size())

                    # error = torch.abs(depth - equi_outputs) * mask
                    # error[error < 0.1] = 0
                    
                    
                    '''
                    #compute_eval_metrics(equi_outputs, depth, mask)
                    rgb_img = img.detach().cpu().numpy()
                    depth_prediction = equi_outputs.detach().cpu().numpy()
                    equi_gt = depth.detach().cpu().numpy()
                    #error_img = error.detach().cpu().numpy()
                    depth_prediction[depth_prediction > 8] = 0
                    
                    # save raw 3D point cloud reconstruction as ply file
                    coords = np.stack(np.meshgrid(range(w), range(h)), -1)
                    coords = np.reshape(coords, [-1, 2])
                    coords += 1
                    uv = coords2uv(coords, w, h)          
                    xyz = uv2xyz(uv) 
                    xyz = torch.from_numpy(xyz).to(img.device)
                    xyz = xyz.unsqueeze(0).repeat(bs, 1, 1)
                    gtxyz = xyz * depth.reshape(bs, w*h, 1)    
                    predxyz = xyz * equi_outputs.reshape(bs, w*h, 1)
                    gtxyz = gtxyz.detach().cpu().numpy()
                    predxyz = predxyz.detach().cpu().numpy()
                    #error = error.detach().cpu().numpy()

                    #if batch_idx % 20 == 0:
                    rgb_img = rgb_img[0, :, :, :].transpose(1, 2, 0)
                    depth_pred_img = depth_prediction[0, 0, :, :]
                    print(depth_pred_img)
                    print(depth_pred_img.max())

                    print(depth_pred_img.shape)
                    depth_pred_img_small = cv2.resize(depth_pred_img, (512, 256), interpolation = cv2.INTER_AREA)
                    print(depth_pred_img_small.shape)


                    depth_gt_img = equi_gt[0, 0, :, :]
                    error_img = error_img[0, 0, :, :] 
                    gtxyz_np = predxyz[0, ...]
                    predxyz_np = predxyz[0, ...]
                    batch_idx = 0
                    cv2.imwrite('{}/test_equi_rgb_{}.png'.format(result_view_dir, batch_idx),
                                rgb_img*255)
                    plot.imsave('{}/test_equi_pred_{}.png'.format(result_view_dir, batch_idx),
                                depth_pred_img, cmap="jet")
                    plot.imsave('{}/test_equi_pred_small_{}.png'.format(result_view_dir, batch_idx),
                                depth_pred_img_small, cmap="jet")                
                    plot.imsave('{}/test_equi_gt_{}.png'.format(result_view_dir, batch_idx),
                                depth_gt_img, cmap="jet")
                    plot.imsave('{}/test_error_{}.png'.format(result_view_dir, batch_idx),
                                error_img, cmap="jet")
                    rgb_img = np.reshape(rgb_img*255, (-1, 3)).astype(np.uint8)
                    #write_ply('{}/test_gt_{}'.format(result_view_dir, batch_idx), [gtxyz_np, rgb_img], ['x', 'y', 'z', 'blue', 'green', 'red'])
                    #write_ply('{}/test_pred_{}'.format(result_view_dir, batch_idx), [predxyz_np, rgb_img], ['x', 'y', 'z', 'blue', 'green', 'red'])
                    '''



            #num_render_view = len(baseline)

            render = dibr_vertical(coarse_depth_pred.clamp(0.1, 8.0), rgb, uvgrid, sgrid, baseline=baseline)

            depth_np = coarse_depth_pred.detach().cpu().numpy()
            if batch_idx % 20 == 0 and batch_idx > 0:
                    depth_coarse_img = depth_np[0, 0, :, :]

                    #depth_coarse_img[depth_coarse_img>8] = 0 #!!!!
                    depth_coarse_img[depth_coarse_img < 0.1] = 0.1
                    depth_coarse_img[depth_coarse_img > 8] = 8                    

                    plot.imsave('{}/test_coarse_depth_{}.png'.format(result_view_dir, batch_idx), depth_coarse_img, cmap="jet")

            val_output = val(rgb, render, depth.squeeze(1), depth_mask.squeeze(1), batch_idx)
            compute_eval_metrics1(coarse_depth_pred, depth, depth_mask)
            compute_eval_metrics2(val_output, depth.squeeze(1), depth_mask.squeeze(1))
	        #-------------------------------------------------------------
            # Loss ---------------------------------
            #total_val_loss += val_loss
            #---------------------------------------
            # Step ------
            global_val+=1
            #------------
        #writer.add_scalar('total validation loss',total_val_loss/(len(val_dataloader)),epoch) #tensorboardX for validation in epoch        
        #writer.add_scalar('total validation crop 26 depth rmse',total_val_crop_rmse/(len(val_dataloader)),epoch) #tensorboardX rmse for validation in epoch
        print('Epoch: {}\n'
        '  Avg. Abs. Rel. Error: coarse {:.4f}, final {:.4f}\n'
        '  Avg. Sq. Rel. Error: coarse {:.4f}, final {:.4f}\n'
        '  Avg. Lin. RMS Error: coarse {:.4f}, final {:.4f}\n'
        '  Avg. Log RMS Error: coarse {:.4f}, final {:.4f}\n'
        '  Inlier D1: coarse {:.4f}, final {:.4f}\n'
        '  Inlier D2: coarse {:.4f}, final {:.4f}\n'
        '  Inlier D3: coarse {:.4f}, final {:.4f}\n\n'.format(
        epoch, 
        abs_rel_error_meter_coarse.avg,
        abs_rel_error_meter.avg,
        sq_rel_error_meter_coarse.avg,
        sq_rel_error_meter.avg,
        math.sqrt(lin_rms_sq_error_meter_coarse.avg),
        math.sqrt(lin_rms_sq_error_meter.avg),
        math.sqrt(log_rms_sq_error_meter_coarse.avg),
        math.sqrt(log_rms_sq_error_meter.avg),
        d1_inlier_meter_coarse.avg,
        d1_inlier_meter.avg,
        d2_inlier_meter_coarse.avg,
        d2_inlier_meter.avg,
        d3_inlier_meter_coarse.avg,
        d3_inlier_meter.avg))

        row = [epoch, '{:.4f}'.format(abs_rel_error_meter.avg.item()), 
            '{:.4f}'.format(sq_rel_error_meter.avg.item()), 
            '{:.4f}'.format(torch.sqrt(lin_rms_sq_error_meter.avg).item()),
            '{:.4f}'.format(torch.sqrt(log_rms_sq_error_meter.avg).item()), 
            '{:.4f}'.format(d1_inlier_meter.avg.item()), 
            '{:.4f}'.format(d2_inlier_meter.avg.item()), 
            '{:.4f}'.format(d3_inlier_meter.avg.item())]
        with open(csv_filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)  
            csvwriter.writerow(row)


    writer.close()
    # End Training
    print("Training Ended hahahaha!!!")
    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
#----------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
        