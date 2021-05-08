import torch
from .base_model import BaseModel
from . import networks
from torchvision.models.vgg import vgg16
from saliency_network import saliency_network_resnet18
from saliency_sampler import Saliency_Sampler


class Pix2PixMaskModel(BaseModel):
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
        self.device_ids = [0,1,2,3]
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_contex','TV'] #,'D_real0','D_real1','D_fake' ,'D_fake1','D_fake2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_B', 'fake_B', 'fake_B_new','comb_B','mask']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(6, 1, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

#        saliency_network = saliency_network_resnet18()
#        task_input_size = 256
#        saliency_input_size = 256
#        self.netG=Saliency_Sampler(self.netG,saliency_network,task_input_size,saliency_input_size,self.device)

#        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
#            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
#                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            # self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.TV = networks.TVLoss().to(self.device)
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
#            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
#            self.optimizers.append(self.optimizer_D)

    def set_input(self, input,model0,model1,opt):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.model1=model1
        self.model1.set_input(input)
        # self.model1.netG=torch.nn.DataParallel(self.model1.netG)
        self.model0=model0
        self.model0.set_input(input)
        # self.model0.netG=torch.nn.DataParallel(self.model0.netG)
        # self.set_requires_grad(self.model0.netG, False)
        # self.set_requires_grad(self.model1.netG, False)
        # with torch.no_grad():

        self.optimizer_G0 = torch.optim.Adam(self.model0.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # # self.optimizer_D0 = torch.optim.Adam(self.model0.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G0)
        # # self.optimizers.append(self.optimizer_D0)

        self.optimizer_G1 = torch.optim.Adam(self.model1.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # # self.optimizer_D1 = torch.optim.Adam(self.model1.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G1)
        # self.optimizers.append(self.optimizer_D1)

        self.netG = torch.nn.DataParallel(self.netG)
        self.model1.netG=torch.nn.DataParallel(self.model1.netG)
        self.model0.netG=torch.nn.DataParallel(self.model0.netG)
            
        AtoB = self.opt.direction == 'AtoB'
#        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self,p):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.model1.forward(p)
        self.fake_B=self.model1.fake_B
        self.tort_A=self.model1.tort_A
        self.model0.forward(self.tort_A,self.fake_B)
        # self.tort_A=self.model1.tort_A.detach()
        # self.fake_B=self.model0.fake_B #.detach()
        self.fake_B_new=self.model0.fake_B_new #.detach()
        self.mask = self.netG(torch.cat((self.fake_B,self.fake_B_new),1))  # G(A)
        self.comb_B = self.fake_B*(1+self.mask)/2+self.fake_B_new*(1-(1+self.mask)/2)

    # def backward_D(self):
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake; stop backprop to the generator by detaching fake_B
    #     fake_AB = torch.cat((self.tort_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     pred_fake = self.model1.netD(fake_AB.detach())
    #     self.loss_D_fake = self.criterionGAN(pred_fake, False,10)
    #     # Real
    #     real_AB = torch.cat((self.tort_A, self.real_B), 1)
    #     pred_real = self.model1.netD(real_AB)
    #     self.loss_D_real1 = self.criterionGAN(pred_real, True,10)
    #     # combine loss and calculate gradients
    #     self.loss_D1 = (self.loss_D_fake + self.loss_D_real1) * 0.5
    #     self.loss_D1.backward()

    #     fake_AB1 = torch.cat((self.tort_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     pred_fake1 = self.model0.netD(fake_AB1.detach())
    #     self.loss_D_fake1 = self.criterionGAN(pred_fake1, False,1)

    #     fake_AB2 = torch.cat((self.tort_A, self.fake_B_new), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     pred_fake2 = self.model0.netD(fake_AB2.detach())
    #     self.loss_D_fake2 = self.criterionGAN(pred_fake2, False,1)
    #     # Real
    #     real_AB = torch.cat((self.tort_A, self.real_B), 1)
    #     pred_real = self.model0.netD(real_AB)
    #     self.loss_D_real0 = self.criterionGAN(pred_real, True,1)
    #     # combine loss and calculate gradients
    #     self.loss_D0 = (self.loss_D_fake1 +self.loss_D_fake2 + self.loss_D_real0) * 0.5
    #     self.loss_D0.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
#        fake_AB = torch.cat((self.tort_A, self.fake_B), 1)
#        pred_fake = self.netD(fake_AB)
#        self.loss_G_GAN = self.criterionGAN(pred_fake, True,0)
        # Second, G(A) = B
        #print(self.opt.lambda_L1)
        self.loss_TV = self.TV(self.mask) * 0
        
        # self.loss_G_L1 = self.criterionL1(self.comb_B, self.real_B) *30

        # self.loss_G_perceptual = self.mse_loss(self.loss_network(self.comb_B), self.loss_network(self.real_B))

        self.loss_G_contex = self.contex(self.comb_B,self.real_B)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_contex + self.loss_TV
        # self.loss_G = self.loss_G_L1+self.loss_G_perceptual
        self.loss_G.backward()

    def optimize_parameters(self,p,epoch):
        self.forward(p)                   # compute fake images: G(A)
        if epoch%3==0:
            self.set_requires_grad(self.model0.netG, True)
            self.set_requires_grad(self.model1.netG, False)
            self.set_requires_grad(self.netG, False)
            self.optimizer_G0.zero_grad()
            self.backward_G()
            self.optimizer_G0.step()
        elif epoch%3==1:
            self.set_requires_grad(self.model0.netG, False)
            self.set_requires_grad(self.model1.netG, True)
            self.set_requires_grad(self.netG, False)
            self.optimizer_G1.zero_grad()
            self.backward_G()
            self.optimizer_G1.step()
        elif epoch%3==2:
            self.set_requires_grad(self.model0.netG, False)
            self.set_requires_grad(self.model1.netG, False)
            self.set_requires_grad(self.netG, True)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

        return self.model0,self.model1
