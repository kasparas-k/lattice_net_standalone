import yaml

import os
import sys
import time

import numpy as np\
import torch
from tqdm import tqdm

from latticenet_py.lattice.diceloss import GeneralizedSoftDiceLoss
from latticenet_py.lattice.lovasz_loss import LovaszSoftmax
from latticenet_py.lattice.models import *

from latticenet_py.callbacks.callback import *
from latticenet_py.callbacks.viewer_callback import *
from latticenet_py.callbacks.visdom_callback import *
from latticenet_py.callbacks.state_callback import *
from latticenet_py.callbacks.phase import *

config_file = "config.yaml"
with open(config_file, 'r') as f:
    config = yaml.load(f)

torch.manual_seed(0)
# torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.benchmark = True
config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)

# #initialize the parameters used for training
train_params = config['train']
model_params = config['model']


def run():
    first_time=True

    experiment_name="s_10tryingback"

    #torch stuff
    lattice=Lattice.create(config_path, "lattice")

    cb_list = []
    cb_list.append(StateCallback())
    cb = CallbacksGroup(cb_list)


    phases= [
        Phase('train', loader_train, grad=True),
        Phase('test', loader_test, grad=False)
    ]
    #model
    model=LNN(model_params['n_classes'], model_params).to("cuda")
    #create loss function
    #loss_fn=GeneralizedSoftDiceLoss(ignore_index=loader_train.label_mngr().get_idx_unlabeled() )
    loss_fn=LovaszSoftmax(ignore_index=train_params['ignore_label'])
    #class_weights_tensor=model.compute_class_weights(loader_train.label_mngr().class_frequencies(), loader_train.label_mngr().get_idx_unlabeled())
    secondary_fn=torch.nn.NLLLoss(ignore_index=train_params['ignore_label'] )  #combination of nll and dice  https://arxiv.org/pdf/1809.10486.pdf
    scheduler=None

    while True:

        for phase in phases:
            cb.epoch_started(phase=phase)
            cb.phase_started(phase=phase)
            model.train(phase.grad)

            pbar = tqdm(total=phase.loader.nr_samples())
            while ( phase.samples_processed_this_epoch < phase.loader.nr_samples()):

                if(phase.loader.has_data()):
                    cloud=phase.loader.get_cloud()

                    is_training = phase.grad

                    #forward
                    with torch.set_grad_enabled(is_training):
                        cb.before_forward_pass(lattice=lattice) #sets the appropriate sigma for the lattice
                        positions, values, target = prepare_cloud(cloud, model_params) #prepares the cloud for pytorch, returning tensors alredy in cuda

                        TIME_START("forward")
                        pred_logsoftmax, pred_raw =model(lattice, positions, values)
                        TIME_END("forward")
                        loss_dice = 0.5*loss_fn(pred_logsoftmax, target)
                        loss_ce = 0.5*secondary_fn(pred_logsoftmax, target)
                        loss = loss_dice+loss_ce

                        #model.summary()

                        #if its the first time we do a forward on the model we need to create here the optimizer because only now are all the tensors in the model instantiated
                        if first_time:
                            first_time=False
                            optimizer=torch.optim.AdamW(model.parameters(), lr=train_params.lr(), weight_decay=train_params.weight_decay(), amsgrad=True)
                            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)
                            if train_params.dataset_name()=="semantickitti":
                                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3)

                            #sanity check that the lattice has enough vertices
                            # sanity_check(lattice)


                        cb.after_forward_pass(pred_softmax=pred_logsoftmax, target=target, cloud=cloud, loss=loss.item(), loss_dice=loss_dice.item(), phase=phase, lr=optimizer.param_groups[0]["lr"]) #visualizes the prediction
                        pbar.update(1)

                    #backward
                    if is_training:
                        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                            scheduler.step(phase.epoch_nr + float(phase.samples_processed_this_epoch) / phase.loader.nr_samples() )
                        optimizer.zero_grad()
                        cb.before_backward_pass()
                        loss.backward()

                        # model.summary()
                        # exit()

                        cb.after_backward_pass()
                        optimizer.step()

                    # Profiler.print_all_stats()

                if phase.loader.is_finished():
                    pbar.close()
                    if not is_training: #we reduce the learning rate when the test iou plateus
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(phase.loss_acum_per_epoch) #for ReduceLROnPlateau
                    cb.epoch_ended(phase=phase, model=model, save_checkpoint=train_params.save_checkpoint(), checkpoint_path=train_params.checkpoint_path() )
                    cb.phase_ended(phase=phase)

def main():
    run()


if __name__ == "__main__":
     main()
