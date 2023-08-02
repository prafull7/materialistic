import os
import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies.ddp import DDPStrategy

import src.utils.args as parse_args
import src.utils.utils as utils
import src.data.synthetic_dataset as synthetic_dataset
import src.models.model_utils_pl as model_utils

import faulthandler
faulthandler.enable()

args, conf = parse_args.parse_args()

if args.resume:
    # load args and conf from checkpoint
    args, conf = utils.get_config_from_checkpoint(args.checkpoint_dir)
    args.resume = True
else:
    print("Saving config to: ", args.checkpoint_dir)
    # create new directories
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    # save args and conf to checkpoint_dir
    utils.save_config_to_checkpoint(args.checkpoint_dir, args, conf)

# Create a tensorboard writer
print("Logging to: ", args.log_dir)
tb_logger = pl_loggers.TensorBoardLogger(args.log_dir, name=args.exp_name)

###### Data ######
# Function to change the random seed for all workers
def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)

BATCH_SIZE = args.batch_size
H, W = args.image_size, args.image_size

train_dataset = synthetic_dataset.SyntheticMaterialDataset(
    args.data_dir, stage="train", use_chroma=args.use_chroma,
    use_depth=args.use_depth, use_normal=args.use_normal, size=(H, W))
val_dataset = synthetic_dataset.SyntheticMaterialDataset(
    args.data_dir, stage="val", use_chroma=args.use_chroma,
    use_depth=args.use_depth, use_normal=args.use_normal, size=(H, W))

print("Length of train dataset: ", len(train_dataset))
print("Length of val dataset: ", len(val_dataset))

train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=BATCH_SIZE,
                                                num_workers=64,
                                                pin_memory=True,
                                                shuffle=True,
                                                worker_init_fn=worker_init_fn,
                                                drop_last=True)

val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=BATCH_SIZE,
                                              num_workers=16,
                                              pin_memory=True,
                                              shuffle=True,
                                              worker_init_fn=worker_init_fn,
                                              drop_last=True)

# model
net = model_utils.create_model(conf, args)

# checkpoint callback to save model to checkpoint_dir
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.checkpoint_dir,
                                                    filename="model-{epoch:02d}",
                                                    save_top_k=1)

# Write pytorch lightning trainer
if args.resume:
    checkpoint_files = os.listdir(args.checkpoint_dir)
    checkpoint_files = [f for f in checkpoint_files if 'ckpt' in f]
    checkpoint_files.sort()
    trainer = pl.Trainer(accelerator="gpu", 
                            devices=args.num_gpus,
                            val_check_interval= args.print_every,
                            limit_val_batches=1,
                            strategy=DDPStrategy(find_unused_parameters=True), 
                            precision=args.precision, 
                            max_epochs=args.epochs,
                            logger=tb_logger,
                            log_every_n_steps=args.print_every,
                            callbacks=[checkpoint_callback],
                            check_val_every_n_epoch=None,
                            gradient_clip_val=0.5, 
                            gradient_clip_algorithm="value",
                            resume_from_checkpoint=os.path.join(args.checkpoint_dir, checkpoint_files[-1]))
else:
    trainer = pl.Trainer(accelerator="gpu", 
                            devices=args.num_gpus,
                            val_check_interval= args.print_every,
                            limit_val_batches=1,
                            strategy=DDPStrategy(find_unused_parameters=True), 
                            precision=args.precision, 
                            max_epochs=args.epochs,
                            logger=tb_logger,
                            log_every_n_steps=args.print_every,
                            check_val_every_n_epoch=None,
                            gradient_clip_val=0.5, 
                            gradient_clip_algorithm="value",
                            callbacks=[checkpoint_callback])

trainer.fit(net, train_data_loader, val_data_loader)
