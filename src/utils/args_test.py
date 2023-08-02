import os
import sys
import argparse
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(description='Parse arguments for training the model.')

    # Experiment arguments
    parser.add_argument('--config', type=str, default="./configs/transformer.cfg", help='Path to the config file.')
    parser.add_argument('--exp_name', type=str, default='materialistic_checkpoint', 
                        help='Name of the experiment.')
    parser.add_argument('--log_dir', type=str, default='../tensorboard/',
                        help='Path to the save directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                        help='Path to the checkpoint directory.')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default= '../data/materialistic_real_dataset/',
                        help='Path to the data directory.')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the images.')
    parser.add_argument('--use_depth', action='store_true', default=False, help="Whether to use depth maps.")
    parser.add_argument('--use_normal', action='store_true', default=False, help="Whether to use normal maps.")
    parser.add_argument('--use_chroma', action='store_true', default=False, help="Whether to use chroma maps.")
    parser.add_argument('--exclude_lightness', action='store_true', default=False, help="Whether to use chroma maps.")

    # Test arguments
    parser.add_argument('--results_dir', type=str, default="./results/", help="results directory")
    parser.add_argument('--method_name', type=str, default="materialistic_checkpoint", help="results directory")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='unet', help='Name of the model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--print_every', type=int, default=100, help='Print loss every n iterations.')

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, args.exp_name)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    cfg = OmegaConf.load(args.config)
    return args, cfg
