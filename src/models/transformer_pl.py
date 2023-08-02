# import sys
# sys.path.append('../')
import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import pytorch_lightning as pl
import src.models.model_utils as model_utils
import src.loss_functions.loss_utils as loss_utils
import src.utils.utils as utils
from omegaconf import DictConfig


class Transformer_pl(pl.LightningModule):
    def __init__(self, conf: DictConfig, args):
        super().__init__()
        print("Making pytorch lightning model")
        self.use_chroma = args.use_chroma
        self.use_depth = args.use_depth
        self.use_normal = args.use_normal
        self.lr = args.lr
        self.print_freq = args.print_every

        self.net = model_utils.create_model(conf)
        self.criterion = loss_utils.create_loss_function(conf)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        image = batch['images']
        input = image
        mat_labels = batch["mat_labels"]
        reference_locations = batch["reference_locations"]
        scores, path1, path2, path3, path4, context_embeddings_1, context_embeddings_2, context_embeddings_3, context_embeddings_4, layer_1, layer_2, layer_3, layer_4 = self.net(input, reference_locations)
        loss = self.criterion(scores, mat_labels)

        # print("Loss has been computed")
        if batch_idx % self.print_freq == 0:
            self.log("train_loss", loss.item(), on_step=True, on_epoch=False)
            # create a visualization using utils.get_combined_visualization
            viz_data = {"images": image,
                        "mat_labels": mat_labels,
                        "reference_locations": reference_locations,
                        "scores": scores}
            viz_data["path1"] = path1
            viz_data["context_embeddings_1"] = context_embeddings_1
            viz_data["layer_1"] = layer_1

            viz_np, path1_viz, path2_viz, path3_viz, path4_viz, context_embeddings_1_viz, context_embeddings_2_viz, context_embeddings_3_viz, context_embeddings_4_viz, layer_1_viz, layer_2_viz, layer_3_viz, layer_4_viz = utils.get_classification_visualization(viz_data)
            self.logger.experiment.add_image("train_images", viz_np, self.global_step)
            self.logger.experiment.add_image("train_path1", path1_viz, self.global_step)
            self.logger.experiment.add_image("train_context_embeddings_1", context_embeddings_1_viz, self.global_step)
            self.logger.experiment.add_image("train_layer_1", layer_1_viz, self.global_step)
            
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['images']
        input = image        
        mat_labels = batch["mat_labels"]
        reference_locations = batch["reference_locations"]
        scores, path1, path2, path3, path4, context_embeddings_1, context_embeddings_2, context_embeddings_3, context_embeddings_4, layer_1, layer_2, layer_3, layer_4 = self.net(input, reference_locations)
        
        loss = self.criterion(scores, mat_labels)
        self.log("val_loss", loss.item(), on_step=True, on_epoch=False)
        
        # create a visualization using utils.get_combined_visualization
        viz_data = {"images": image,
                    "mat_labels": mat_labels,
                    "reference_locations": reference_locations,
                    "scores": scores}
            
        viz_data["path1"] = path1
        viz_data["context_embeddings_1"] = context_embeddings_1
        viz_data["layer_1"] = layer_1
        viz_np, path1_viz, path2_viz, path3_viz, path4_viz, context_embeddings_1_viz, context_embeddings_2_viz, context_embeddings_3_viz, context_embeddings_4_viz, layer_1_viz, layer_2_viz, layer_3_viz, layer_4_viz = utils.get_classification_visualization(viz_data)
        self.logger.experiment.add_image("val_images", viz_np, self.global_step)
        self.logger.experiment.add_image("val_path1", path1_viz, self.global_step)
        self.logger.experiment.add_image("val_context_embeddings_1", context_embeddings_1_viz, self.global_step)
        self.logger.experiment.add_image("val_layer_1", layer_1_viz, self.global_step)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def load_checkpoint(self, checkpoint_path, map_location=None):
        files = os.listdir(checkpoint_path)
        print(files)
        
        if map_location is None:
            map_location = torch.device('cuda:0')
        print("map_location", map_location)
        files = [f for f in files if ".ckpt" in f] + [f for f in files if "model.pth" in f]
        if '.ckpt' in files[-1]:
            print(files[-1])
            print(torch.load(os.path.join(checkpoint_path, files[-1]), map_location=map_location).keys())
            self.load_state_dict(torch.load(os.path.join(checkpoint_path, files[-1]), map_location= torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))['state_dict'])
        else:
            state_dict = torch.load(os.path.join(checkpoint_path, files[-1]), map_location=map_location)     
            # change the prefix of the keys to match the prefix of the model
            for key in list(state_dict.keys()):
                state_dict['net.' + key.replace("module.", '')] = state_dict.pop(key)

            self.load_state_dict(state_dict)