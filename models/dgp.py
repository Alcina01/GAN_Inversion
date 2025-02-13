import os
from copy import deepcopy

import numpy as np
import cv2
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from PIL import Image
from skimage import color
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.autograd import Variable
import torch.nn as nn

import models
import utils
from models.downsampler import Downsampler
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



class DGP(object):

    def __init__(self, config):
        self.rank, self.world_size = 0, 1
        if config['dist']:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.config = config
        self.mode = config['dgp_mode']
        self.update_G = config['update_G']
        self.update_embed = config['update_embed']
        self.iterations = config['iterations']
        self.ftr_num = config['ftr_num']
        self.ft_num = config['ft_num']
        self.lr_ratio = config['lr_ratio']
        self.G_lrs = config['G_lrs']
        self.z_lrs = config['z_lrs']
        self.use_in = config['use_in']
        self.select_num = config['select_num']
        self.factor = 2 if self.mode == 'hybrid' else 4  # Downsample factor
        self.mask= None
        self.maskpair = None

        # create model
        self.G = models.Generator(**config).cuda()
        self.D = models.Discriminator(
            **config).cuda() if config['ftr_type'] == 'Discriminator' else None
        self.G.optim = torch.optim.Adam(
            [{'params': self.G.get_params(i, self.update_embed)}
                for i in range(len(self.G.blocks) + 1)],
            lr=config['G_lr'],
            betas=(config['G_B1'], config['G_B2']),
            weight_decay=0,
            eps=1e-8)

        # load weights
        if config['random_G']:
            self.random_G()
        else:
            utils.load_weights(
                self.G if not (config['use_ema']) else None,
                self.D,
                config['weights_root'],
                name_suffix=config['load_weights'],
                G_ema=self.G if config['use_ema'] else None,
                strict=False)

        self.G.eval()
        if self.D is not None:
            self.D.eval()
        self.G_weight = deepcopy(self.G.state_dict())

        # prepare latent variable and optimizer
        self._prepare_latent()
        # prepare learning rate scheduler
        self.G_scheduler = utils.LRScheduler(self.G.optim, config['warm_up'])
        self.z_scheduler = utils.LRScheduler(self.z_optim, config['warm_up'])

        # loss functions
        self.mse = torch.nn.MSELoss()
        if config['ftr_type'] == 'Discriminator':
            self.ftr_net = self.D
            self.criterion = utils.DiscriminatorLoss(ftr_num=config['ftr_num'][0])
        else:
            vgg = torchvision.models.vgg16(pretrained=True).cuda().eval()
            self.ftr_net = models.subsequence(vgg.features, last_layer='20')
            self.criterion = utils.PerceptLoss()

        # Downsampler for producing low-resolution image
        self.downsampler = Downsampler(
            n_planes=3,
            factor=self.factor,
            kernel_type='lanczos2',
            phase=0.5,
            preserve_size=True).type(torch.cuda.FloatTensor)

    def _prepare_latent(self):
        # import ipdb;ipdb.set_trace
        num_latents = 100  # Define the size of the latent space
        self.z = torch.nn.Parameter(torch.randn(num_latents, self.G.dim_z).cuda())  # Learnable latent space
        print("resetting the latent space")
        # self.z = torch.zeros((1, self.G.dim_z)).normal_().cuda()

        # self.z = Variable(self.z, requires_grad=True)
        self.z_optim = torch.optim.Adam(
            [{'params': self.z, 'lr': self.z_lrs[0]}],  # Optimizing self.z with learning rate self.z_lrs[0]
            betas=(self.config['G_B1'], self.config['G_B2']),  # Adam momentum terms (β1, β2)
            weight_decay=0,  # No L2 regularization
            eps=1e-8  # Numerical stability
        )
        self.y = torch.zeros(1).long().cuda()

    def reset_G(self):
        self.G.load_state_dict(self.G_weight, strict=False)
        self.G.reset_in_init()
        if self.config['random_G']:
            self.G.train()
        else:
            self.G.eval()

    def random_G(self):
        self.G.init_weights()

    def set_target(self, target, category, img_path,mask,maskpair,gt,i):
        # self.target_origin = target
        self.target_origin = gt
        self.maskpair =maskpair
        self.target_origin = self.target_origin.cuda()
        self.mask =mask
        self.target = self.pre_process(target, True)
        self.y.fill_(category.item())
        self.img_name = img_path[img_path.rfind('/') + 1:img_path.rfind('.')]
        # Sample a latent vector from the global latent space
        self.latent_index = torch.randint(0, self.z.shape[0], (1,)).cuda()  # Choose a latent index
        self.current_z = self.z[self.latent_index]  # Select the latent vector
        index = i
    

    def run(self,i, save_interval=None):
        index =i
        save_imgs = self.target.clone()
        save_imgs2 = save_imgs.cpu().clone()
        loss_dict = {}
        curr_step = 0
        count = 0
        for stage, iteration in enumerate(self.iterations):
            # setup the number of features to use in discriminator
            self.criterion.set_ftr_num(self.ftr_num[stage])
            print("self z : \n",self.z)

            for i in range(iteration):
                curr_step += 1
                # setup learning rate
                self.G_scheduler.update(curr_step, self.G_lrs[stage],
                                        self.ft_num[stage], self.lr_ratio[stage])
                self.z_scheduler.update(curr_step, self.z_lrs[stage])
                self.z_optim.zero_grad()
                # import ipdb;ipdb.set_trace()
                if self.update_G:
                    self.G.optim.zero_grad()
                # apply degradation transform
                x = self.G(self.current_z, self.G.shared(self.y), use_in=self.use_in[stage])
                x_map = self.pre_process(x, False)

                # calculate losses in the degradation space
                ftr_loss = self.criterion(self.ftr_net, x_map, self.target)
                mse_loss = self.mse(x_map, self.target)
                # nll corresponds to a negative log-likelihood loss
                nll = self.z**2 / 2
                nll = nll.mean()
                l1_loss = F.l1_loss(x_map, self.target)

                


                # Combining losses -----------------------------------------------------
                # print(self.target_origin.shape)
                # Convert tensors to HWC format and move to CPU
                # if i == 199:
                #     img_tensor = self.target_origin.squeeze(0).permute(1, 2, 0).cpu().detach()
                #     img_tensor2 = x_map.squeeze(0).permute(1, 2, 0).cpu().detach()
                #     img_tensor3 = (self.target_origin * (1-self.mask)).squeeze(0).permute(1, 2, 0).cpu().detach() #this
                #     img_tensor4 = (x * (1-self.mask)).squeeze(0).permute(1, 2, 0).cpu().detach()     # this 
                #     img_tensor5 = (1-self.mask).squeeze(0).permute(1, 2, 0).cpu().detach()
                #     img_tensor6 = ( self.maskpair).squeeze(0).permute(1, 2, 0).cpu().detach()
                #     img_tensor7 = (self.target).squeeze(0).permute(1, 2, 0).cpu().detach()
                #     img_tensor8 = (self.target * (1-self.mask)).squeeze(0).permute(1, 2, 0).cpu().detach() 

                #     # Normalize images to [0,1]
                #     img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
                #     img_tensor2 = (img_tensor2 - img_tensor2.min()) / (img_tensor2.max() - img_tensor2.min())
                #     img_tensor3 = (img_tensor3 - img_tensor3.min()) / (img_tensor3.max() - img_tensor3.min())
                #     img_tensor4 = (img_tensor4 - img_tensor4.min()) / (img_tensor4.max() - img_tensor4.min())
                #     img_tensor5 = (img_tensor5 - img_tensor5.min()) / (img_tensor5.max() - img_tensor5.min())
                #     img_tensor6 = (img_tensor6 - img_tensor6.min()) / (img_tensor6.max() - img_tensor6.min())
                #     img_tensor7 = (img_tensor7 - img_tensor7.min()) / (img_tensor7.max() - img_tensor7.min())
                #     img_tensor8 = (img_tensor8 - img_tensor8.min()) / (img_tensor8.max() - img_tensor8.min())

                #     # Convert tensors to NumPy
                #     img_numpy1 = img_tensor.numpy()
                #     img_numpy2 = img_tensor2.numpy()
                #     img_numpy3 = img_tensor3.numpy()
                #     img_numpy4 = img_tensor4.numpy()
                #     img_numpy5 = img_tensor5.numpy()
                #     img_numpy6 = img_tensor6.numpy()
                #     img_numpy7 = img_tensor7.numpy()
                #     img_numpy8 = img_tensor8.numpy()

                #     # Create figure for 2x3 image display
                #     fig, axes = plt.subplots(2, 4, figsize=(18, 12))

                #     # Titles for each subplot
                #     titles = [
                #         "Ground Truth", "Generated Image", 
                #         "Ground Truth + Mask", "Generated Image (B4 Preprocessing)+ Mask", 
                #         "Mask", "GT-Mask" , "Input Image" , "Input Image + Mask"
                #     ]
                #     images = [img_numpy1, img_numpy2, img_numpy3, img_numpy4, img_numpy5, img_numpy6,img_numpy7, img_numpy8]

                #     # Loop through images and plot them
                #     for i, ax in enumerate(axes.flat):
                #         ax.imshow(images[i])
                #         ax.axis("off")
                #         ax.set_title(titles[i])

                #     # Adjust layout and show plot
                #     plt.tight_layout()
                #     plt.show()

                #Creating BB fro masks -----------------------------
                combined_tensor_mask= self.mask.logical_or(self.maskpair)
                target_np = (self.target_origin.detach().cpu().numpy()[0] + 1) / 2
                x_np = (x.detach().cpu().numpy()[0] + 1) / 2
                target_np = np.transpose(target_np, (1, 2, 0))
                x_np = np.transpose(x_np, (1, 2, 0))
                mask_np = (1-self.mask).detach().cpu().numpy()[0]  
                mask_np =np.transpose(mask_np,(1,2,0))
                mask_np =np.mean(mask_np,axis=2)

                gt_mask_np = self.maskpair.detach().cpu().numpy()[0]  
                gt_mask_np =np.transpose(gt_mask_np,(1,2,0))
                gt_mask_np =np.mean(gt_mask_np,axis=2)

                # combined_mask_np = np.ma.mask_or(mask_np ,gt_mask_np)
                combined_mask_np = np.clip(mask_np + gt_mask_np, 0, 1)

                # import ipdb; ipdb.set_trace()
                y_indices_gt, x_indices_gt = np.where(combined_mask_np > 0)
                xmin, xmax = x_indices_gt.min(), x_indices_gt.max()
                ymin, ymax = y_indices_gt.min(), y_indices_gt.max()

                cropped_image_mask = combined_mask_np[ymin:ymax+1, xmin:xmax+1]
                cropped_image = target_np[ymin:ymax+1, xmin:xmax+1, :]
                cropped_x_np = x_np[ymin:ymax+1, xmin:xmax+1, :]

                #--------------------------------------------------
                x_map_np =(x_map.detach().cpu().numpy()[0] + 1) / 2
                x_map_np = np.transpose(x_map_np, (1, 2, 0))
                full_ssim = compare_ssim(x_map_np, target_np, win_size=3, channel_axis=2, data_range=1.0)

                full_loss = ftr_loss * self.config['w_D_loss'][stage] + \
                    mse_loss * self.config['w_mse'][stage]  + (full_ssim *2)

                ssim = compare_ssim(cropped_image, cropped_x_np, win_size=3, channel_axis=2, data_range=1.0)
                masked_mse_loss = self.mse(x[combined_tensor_mask.bool()],  self.target_origin[combined_tensor_mask.bool()])  # before PP
                # print("SSIM from training Loss:",ssim)

                masked_loss = masked_mse_loss * self.config['w_mse'][stage]  *(ssim *5)
                #-----------------------------------------------------
                # loss = full_loss * 0.5 + nll * self.config['w_nll']      # Original Loss
                # loss =masked_loss + nll * self.config['w_nll']                            # Masked area Loss
                # loss = full_loss * 0.7 + masked_loss *0.5 + nll * self.config['w_nll']    # Hybrid Loss
                loss = full_loss

                print("doing backward step")
                loss.backward()
                print('done baclward step')
                self.z_optim.step()
                if self.update_G:
                    self.G.optim.step()

                # These losses are calculated in the [-1,1] image scale
                # We record the rescaled MSE and L1 loss, corresponding to [0,1] image scale
                # loss_dict = {
                #     'ftr_loss': ftr_loss.,
                #     'nll': nll,
                #     'mse_loss': mse_loss / 4,
                #     'l1_loss': l1_loss / 2 ,
                #     'masked_mse_loss' : masked_mse_loss,
                #     'BB_ssim' : ssim,
                #     'full_ssim' : full_ssim

                # }
                loss_dict = {
                    'ftr_loss': 0,
                    'nll': 0,
                    'mse_loss': 0 / 4,
                    'l1_loss': 0 / 2 ,
                    'masked_mse_loss' : 0,
                    'BB_ssim' : 0,
                    'full_ssim' : 0

                }

                metrics, x2 = self.get_metrics(x,i)
                loss_dict = {**loss_dict, **metrics}

                if i == 0 or (i + 1) % self.config['print_interval'] == 0:
                    if self.rank == 0:
                        print(', '.join(
                            ['Stage: [{0}/{1}]'.format(stage + 1, len(self.iterations))] +
                            ['Iter: [{0}/{1}]'.format(i + 1, iteration)] +
                            ['%s : %+4.4f' % (key, loss_dict[key]) for key in loss_dict]
                        ))
                    # save image sheet of the reconstruction process
                    save_imgs = torch.cat((save_imgs, x), dim=0)
                    torchvision.utils.save_image(
                        save_imgs.float(),
                        '%s/images_sheet/%s_%s.jpg' %
                        (self.config['exp_path'], self.img_name, index),
                        nrow=int(save_imgs.size(0)**0.5),
                        normalize=True)

                # if save_interval is not None:
                #     if i == 0 or (i + 1) % save_interval[stage] == 0:
                #         count += 1
                #         save_path = '%s/images/%s' % (self.config['exp_path'],
                #                                       self.img_name)
                #         if not os.path.exists(save_path):
                #             os.makedirs(save_path)
                #         img_path = os.path.join(
                #             save_path, '%s_%03d.jpg' % (self.img_name, count))
                #         utils.save_img(x[0], img_path)

                # stop the reconstruction if the loss reaches a threshold
                if mse_loss.item() < self.config['stop_mse'] or ftr_loss.item(
                ) < self.config['stop_ftr']:
                    break

        # save images
        utils.save_img(
            self.target[0], '%s/images/%s_%s_target.png' %
            (self.config['exp_path'], self.img_name, index))
        utils.save_img(
            self.target_origin[0],
            '%s/images/%s_%s_target_origin.png' %
            (self.config['exp_path'], self.img_name, index))
        utils.save_img(
            x[0], '%s/images/%s_%s.png' %
            (self.config['exp_path'], self.img_name, index))

        if self.config['save_G']:
            torch.save(
                self.G.state_dict(), '%s/G_%s_%s.pth' %
                (self.config['exp_path'], self.img_name, index))
            torch.save(self.z, '%s/z_%s_%s.pth' %
                # self.z, '%s/z_%s_%s.pth' %
                (self.config['exp_path'], self.img_name, index))
        return loss_dict

    def select_z(self, select_y=False):
        print(f"selecting z")
        with torch.no_grad():
            if self.select_num == 0:
                self.z.zero_()
                return
            elif self.select_num == 1:
                self.z.normal_()
                return
            z_all, y_all, loss_all = [], [], []
            if self.rank == 0:
                print('Selecting z from {} samples'.format(self.select_num))
            # only use last 3 discriminator features to compare
            self.criterion.set_ftr_num(3)
            for i in range(self.select_num):
                self.z.normal_(mean=0, std=self.config['sample_std'])
                idx = torch.randint(0, self.z.shape[0], (1,))  # Pick a random latent index
                z_sample = self.z[idx]  # Select a global latent
                z_all.append(self.z.cpu())
                if select_y:
                    self.y.random_(0, self.config['n_classes'])
                    y_all.append(self.y.cpu())
                # x = self.G(self.z, self.G.shared(self.y))
                x = self.G(z_sample.view(1, -1), self.G.shared(self.y))  # Ensure correct sh
                x = self.pre_process(x)
                ftr_loss = self.criterion(self.ftr_net, x, self.target)
                loss_all.append(ftr_loss.view(1).cpu())
                if self.rank == 0 and (i + 1) % 100 == 0:
                    print('Generating {}th sample'.format(i + 1))
            loss_all = torch.cat(loss_all)
            idx = torch.argmin(loss_all)
            self.z.copy_(z_all[idx])
            if select_y:
                self.y.copy_(y_all[idx])
            self.criterion.set_ftr_num(self.ftr_num[0])

    def pre_process(self, image, target=True):
        if self.mask != None:
            image = image * self.mask
        return image

    def get_metrics(self, x,i):
        with torch.no_grad():
            # print(i)
            l1_loss_origin = F.l1_loss(x, self.target_origin) / 2
            mse_loss_origin = self.mse(x, self.target_origin) / 4

            metrics = {
                'l1_loss_origin': l1_loss_origin,
                'mse_loss_origin': mse_loss_origin
            }
            target_np = (self.target_origin.detach().cpu().numpy()[0] + 1) / 2
            x_np = (x.detach().cpu().numpy()[0] + 1) / 2
            target_np = np.transpose(target_np, (1, 2, 0))
            x_np = np.transpose(x_np, (1, 2, 0))

            if self.mode == 'inpainting':
                # mask_np = self.maskpair.detach().cpu().numpy()[0]
                # # mask_np = self.mask.detach().cpu().numpy()[0]
                # mask_np = np.transpose(mask_np, (1, 2, 0))
                # mask_np = ((mask_np > 0)).astype(np.float32)# Ensure the mask is binary
                # masked_target_np = target_np * mask_np# Apply mask to images
                # masked_x_np = x_np * mask_np

                mask_np = (1-self.mask).detach().cpu().numpy()[0]  
                mask_np =np.transpose(mask_np,(1,2,0))
                mask_np =np.mean(mask_np,axis=2)

                gt_mask_np = self.maskpair.detach().cpu().numpy()[0]  
                gt_mask_np =np.transpose(gt_mask_np,(1,2,0))
                gt_mask_np =np.mean(gt_mask_np,axis=2)

                # combined_mask_np = np.ma.mask_or(mask_np ,gt_mask_np)
                combined_mask_np = np.clip(mask_np + gt_mask_np, 0, 1)

                # import ipdb; ipdb.set_trace()
                y_indices_gt, x_indices_gt = np.where(combined_mask_np > 0)
                xmin, xmax = x_indices_gt.min(), x_indices_gt.max()
                ymin, ymax = y_indices_gt.min(), y_indices_gt.max()
                
                cropped_image_mask = combined_mask_np[ymin:ymax+1, xmin:xmax+1]
                cropped_image = target_np[ymin:ymax+1, xmin:xmax+1, :]
                cropped_x_np = x_np[ymin:ymax+1, xmin:xmax+1, :]
                
                masked_target_np = target_np * combined_mask_np[:, :, None]
                masked_x_np = x_np * combined_mask_np[:, :, None]

            

                # Convert masked images for display --------------------------------------------
                # if i == 199:
                #     mask_display = (mask_np * 255).astype(np.uint8)
                #     tg_img = (x_np * 255).astype(np.uint8)
                #     masked_target_display = (masked_target_np * 255).astype(np.uint8)
                #     masked_x_display = (masked_x_np * 255).astype(np.uint8)
                #     cropped_mask_display = (cropped_image_mask * 255).astype(np.uint8)
                #     cropped_masked_target_display = (cropped_image * 255).astype(np.uint8)
                #     cropped_masked_x_display = (cropped_x_np * 255).astype(np.uint8)
                #     target_display = (target_np * 255).astype(np.uint8)
                #     x_display = (x_np * 255).astype(np.uint8)
                #     cv2.imshow("Target Image (Ground Truth)", target_display)
                #     cv2.imshow("Predicted Image", x_display)
                #     cv2.imshow("Occlusion Mask", mask_display)
                #     cv2.imshow("Masked Target Image", masked_target_display)
                #     cv2.imshow("Masked Predicted Image", masked_x_display)
                #     cv2.imshow(" Cropped Mask", cropped_mask_display)
                #     cv2.imshow("Cropped Predicted Image", cropped_masked_x_display)
                #     cv2.imshow(" Cropped Target Image", cropped_masked_target_display)
                #     cv2.imshow(" UN- Cropped Target Image", tg_img)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                #-------------------------------------------------------------------------------

            ssim = compare_ssim(cropped_image, cropped_x_np, win_size=3, channel_axis=2, data_range=1.0)
            psnr = compare_psnr(target_np, x_np)
            metrics['psnr'] = torch.Tensor([psnr]).cuda()
            # metrics['ssim'] = torch.Tensor([ssim]).cuda()

            return metrics, x

