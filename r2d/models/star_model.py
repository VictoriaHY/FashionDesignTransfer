import torch
from .base_model import BaseModel
from . import networks
from torchvision.models.vgg import vgg16
from saliency_network import saliency_network_resnet18
from saliency_sampler import Saliency_Sampler
import numpy as np
import torch.nn.functional as F


class StarModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1','G_perceptual', 'D_real', 'D_fake','D_cls','G_cls']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B','want_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(6, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # saliency_network = saliency_network_resnet18()
        # task_input_size = 256
        # saliency_input_size = 256
        # self.netG=Saliency_Sampler(self.netG,saliency_network,task_input_size,saliency_input_size,self.device)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(6, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,1)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.mse_loss = torch.nn.MSELoss()
            layers = {
                "conv_4_2": 1.0,
                "conv_3_2": 1.0
                }
            self.contex = networks.Contextual_Loss(layers, max_1d_size=64).to(self.device)
            vgg = vgg16(pretrained=True)
            loss_network = torch.nn.Sequential(*list(vgg.features)[:31]).eval().to(self.device)
            for param in loss_network.parameters():
                param.requires_grad = False
            self.loss_network = loss_network
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target)

    def set_input(self, input,input2):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.want_B = input2['B' if AtoB else 'A'].to(self.device)
        self.label = (input['flag']).to(self.device)
        # print(self.label)
        self.labelput = self.label2onehot(input['flag'],4).to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self,p):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.inp = torch.cat((self.real_A,self.want_B), 1)
        self.fake_B= self.netG(self.inp)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # fake_AB = torch.cat((fake_AB, self.fake_B), 1)
        pred_fake,pred_cls = self.netD(fake_AB.detach())
        # print(pred_cls)
        # print(self.label)
        self.loss_D_cls = self.classification_loss(pred_cls,self.label)*20
        self.loss_D_fake = self.criterionGAN(pred_fake, False,1)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # real_AB = torch.cat((real_AB, self.fake_B), 1)
        pred_real,pred_cls2 = self.netD(real_AB.detach())
        # self.loss_D_cls2 = self.classification_loss(pred_cls2,self.label)
        self.loss_D_real = self.criterionGAN(pred_real, True,1)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + self.loss_D_cls #+ self.loss_D_cls2
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # fake_AB = torch.cat((fake_AB, self.fake_B), 1)
        pred_fake,pred_cls = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True,0)
        self.loss_G_cls = self.classification_loss(pred_cls,self.label)*20
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 * 0.3
        # self.loss_G_contex = self.contex(self.fake_B,self.real_B)

        self.loss_G_perceptual = self.mse_loss(self.loss_network(self.fake_B), self.loss_network(self.real_B))*5
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1+self.loss_G_perceptual+self.loss_G_cls
        self.loss_G.backward()

    def optimize_parameters(self,p):
        self.set_requires_grad(self.netG, True)
        self.forward(p)                   # compute fake images: G(A)
        # update D
        # self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        # self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

