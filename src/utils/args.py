import os
import sys
import argparse
from omegaconf import OmegaConf



def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for training the model.')

    # Experiment arguments
    parser.add_argument('--config', type=str, default="./configs/transformer.cfg", help='Path to the config file.')
    parser.add_argument('--exp_name', type=str, default='materialistic_checkpoint_new', 
                        help='Name of the experiment.')
    parser.add_argument('--log_dir', type=str, default='../tensorboard/',
                        help='Path to the save directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                        help='Path to the checkpoint directory.')
    parser.add_argument('--real_data', action='store_true', default=False, help="Pass to train with real data.")
    parser.add_argument('--resume', action='store_true', default=False, help="Whether to resume training from a checkpoint.")
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use.')
    parser.add_argument('--precision', type=int, default=32, help='Precision level to use.')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default= '../data/materialistic_synthetic_dataset/',
                        help='Path to the data directory.')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the images.')

    # Model arguments
    parser.add_argument('--model_name', type=str, default='unet', help='Name of the model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--print_every', type=int, default=500, help='Print loss every n iterations.')
    
    args = parser.parse_args()
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    cfg = OmegaConf.load(args.config)
    return args, cfg
