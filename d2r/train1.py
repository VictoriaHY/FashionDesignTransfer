"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    
    # model = create_model(opt)      # create a model given opt.model and other options
    # model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    

    opt2 = TrainOptions().parse()  # get test options
    # hard-code some parameters for test
    # opt2.dataroot="./data2"
    # opt2.direction="AtoB"
    # opt2.gan_mode="vanilla"
    
    print(opt2.model)

    dataset = create_dataset(opt2)
    dataset_size = len(dataset)
    visualizer = Visualizer(opt2)

    model2 = create_model(opt2)      # create a model given opt.model and other options
    print(opt2.model)
    model2.setup(opt2)
    for epoch in range(opt2.epoch_count, opt2.niter + opt2.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt2.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt2.batch_size
            epoch_iter += opt2.batch_size
            model2.set_input(data)         # unpack data from dataset and apply preprocessing
            flag=1
            if epoch>100:
                flag=1
            model2.optimize_parameters(1)   # calculate loss functions, get gradients, update network weights

            if total_iters % 1 == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt2.update_html_freq == 0
                model2.compute_visuals()
                visualizer.display_current_results(model2.get_current_visuals(), epoch, save_result)

            if total_iters % opt2.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model2.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt2.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt2.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt2.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt2.save_by_iter else 'latest'
                model2.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt2.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model2.save_networks('latest')
            model2.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt2.niter + opt2.niter_decay, time.time() - epoch_start_time))
        model2.update_learning_rate()                     # update learning rates at the end of every epoch.
