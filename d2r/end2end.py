"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model,create_model1
from util.visualizer import save_images
from util import html
import time
from options.train_options import TrainOptions
from util.visualizer import Visualizer
import torch


if __name__ == '__main__':
    

    opt1 = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt1.dataroot="./data"
    opt1.name="h1_pix2pix"
    opt1.model="pix2pix"
    opt1.direction="AtoB"
    # opt1.gan_mode="wgangp"
    opt1.continue_train=True
    # opt1.num_threads = 0   # test code only supports num_threads = 1
    # opt1.batch_size = 1    # test code only supports batch_size = 1
    # opt1.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # opt1.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    # opt1.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #dataset1 = create_dataset(opt1)
    print(opt1.model)
    model1 = create_model(opt1)      # create a model given opt.model and other options
    print(opt1.model)
    model1.setup(opt1)
    # model1=torch.nn.DataParallel(model1)

    opt2 = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt2.dataroot="./data"
    opt2.name="hrawp1_pix2pix"
    opt2.model="pix2pix_rawp"
    opt2.direction="AtoB"
    # opt2.continue_train=True
    # opt2.gan_mode="wgangp"
    # opt2.num_threads = 0   # test code only supports num_threads = 1
    # opt2.batch_size = 1    # test code only supports batch_size = 1
    # opt2.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # opt2.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    # opt2.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #dataset1 = create_dataset(opt1)
    print(opt2.model)
    model2 = create_model(opt2)      # create a model given opt.model and other options
    # print(opt2.model)
    model2.setup(opt2)
    # model2=torch.nn.DataParallel(model2)


    opt = TrainOptions().parse()   # get training options
    #opt.input_nc=6
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    opt.dataroot="./data"
    opt.model="pix2pix_maskp"
    opt.gan_mode="wgangp"
    opt.continue_train=True
    opt.niter=136
    opt.niter_decay=136
    #opt.netG="resnet_9blocks"
    model = create_model1(opt,model1,model2)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    # model=torch.nn.DataParallel(model)



    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            flag=0
            if epoch>-1:
                flag=1
            model.optimize_parameters(flag)   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                # model.model0.save_networks(save_suffix)
                # model.model1.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % 1000 == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            # model.model0.save_networks('latestn')
            # model.model0.save_networks(epoch)
            # model.model1.save_networks('latestn')
            # model.model1.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.


    
