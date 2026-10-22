import os
import configargparse
from typing import Optional
from functools import partial
from torch.utils.data import DataLoader

from .loss_functions import *
from .dataio import *
from .meta_modules import *
from .utils import *
from .training import *
from attacks.hypernet.hypernet_attack import SurrogateClassifier
from utils.wandb_manager import WandbManager
def main(
     #     batch_size=256,
     #     lr=5e-5,
     #     num_epochs=401,
     #     kl_weight=1e-1,
     #     fw_weight=1e2,
     #     epochs_til_ckpt=10,
     #     steps_til_summary=1000,
     #     base_path=None,
         dataset_name,
         *,
         wandb_manager: Optional[WandbManager] = None,
         device='cuda',
         **kwargs,
     ):
     exp_ckpt_path = kwargs.get('exp_ckpt_path', None)
     data_path = kwargs['path_config'].data_path
     coord_dataset = GeneralImages(split='train', dataname=dataset_name, data_path=data_path, )
     # generalization_dataset = ImageGeneralizationWrapper(coord_dataset,
     #                                                        train_sparsity_range=train_sparsity_range,
     #                                                        generalization_mode=gmode)
     image_resolution = coord_dataset.sidelength
     img_channels = coord_dataset.img_channels

     dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=kwargs.get('batch_size', 256), pin_memory=True, num_workers=2)
     model = ConvolutionalNeuralProcessImplicit2DHypernet(in_features=img_channels, # coord
                                                          out_features=img_channels,
                                                          image_resolution=image_resolution,
                                                          hidden_features=kwargs['inr_config'].width,
                                                          num_hidden_layers=kwargs['inr_config'].depth
                                                          )
     model.cuda()
     print(f"model params: {sum([p.numel() for p in model.parameters()]) / 1000000} M  ")

     if os.path.exists(os.path.join(exp_ckpt_path, "final.pth")):
          state_dict = torch.load(os.path.join(exp_ckpt_path, "final.pth"), weights_only=False)
          model.load_state_dict(state_dict)
          print(f"Loaded model from {os.path.join(exp_ckpt_path, 'final.pth')}")
          
     else:
          print(f"No model found at {os.path.join(exp_ckpt_path, 'final.pth')}, thus training from scratch")
               
          loss_fn = partial(image_hypernetwork_loss, None, kwargs['kl_weight'], kwargs['fw_weight'], kwargs['cls_weight'], kwargs['victim_wrapped'])

          model = train(model=model, 
                         train_dataloader=dataloader, 
                         epochs=kwargs['epochs'], 
                         lr=kwargs['lr'],
                         steps_til_summary=kwargs.get('steps_til_summary', 1000), 
                         epochs_til_checkpoint=kwargs.get('epochs_til_ckpt', 10),
                         checkpoints_dir=exp_ckpt_path, 
                         loss_fn=loss_fn, 
                         wandb_manager=wandb_manager, 
                         builder=kwargs['builder'],
                         clip_grad=True,
                         exp_plot_dir=kwargs.get('exp_plot_path', None),)
     model.eval()
     model = SurrogateClassifier(model, kwargs['victim_wrapped']).to(device)

     return model

if __name__ == "__main__":
     p = configargparse.ArgumentParser()
     p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

     p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
     p.add_argument('--experiment_name', type=str, required=True,
                    help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

     # General training options
     p.add_argument('--batch_size', type=int, default=100)
     p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5')
     p.add_argument('--num_epochs', type=int, default=401,
                    help='Number of epochs to train for.')
     p.add_argument('--kl_weight', type=float, default=1e-1,
                    help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')
     p.add_argument('--fw_weight', type=float, default=1e2,
                    help='Weight for the l2 loss term on the weights of the sine network')
     p.add_argument('--train_sparsity_range', type=int, nargs='+', default=[10, 200],
                    help='Two integers: lowest number of sparse pixels sampled followed by highest number of sparse'
                         'pixels sampled when training the conditional neural process')

     p.add_argument('--epochs_til_ckpt', type=int, default=10,
                    help='Time interval in seconds until checkpoint is saved.')
     p.add_argument('--steps_til_summary', type=int, default=1000,
                    help='Time interval in seconds until tensorboard summary is saved.')

     p.add_argument('--dataset_name', type=str, default='cifar',
                    help='Time interval in seconds until tensorboard summary is saved.')
     p.add_argument('--model_type', type=str, default='sine',
                    help='Nonlinearity for the hypo-network module')

     p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

     p.add_argument('--conv_encoder', action='store_true', default=False, help='Use convolutional encoder process')
     opt = p.parse_args()

     main(**opt)
