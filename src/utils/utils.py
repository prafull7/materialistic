import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import numpy as np
import torch
import cv2
import pickle
import kornia.color.lab as lab
import torchvision

def detach_all(t: torch.Tensor):
    return t.detach().cpu()

def save_config_to_checkpoint(checkpoint_path, args, conf):
    configuration = {"args": args,
                     "conf": conf}
    pickle.dump(configuration, open(checkpoint_path + "/args.pkl", "wb"))

def get_config_from_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path + "/args.pkl"):
        configuration = pickle.load(open(checkpoint_path + "/args.pkl", "rb"))
        return configuration["args"], configuration["conf"]
    return None

def save_cur_batch_to_checkpoint(checkpoint_path, cur_batch):
    torch.save(cur_batch, checkpoint_path + "/cur_batch.pt")

def get_cur_batch_from_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path + "/cur_batch.pt")

def normalize(x):
    return (x - x.min())/(x.max() - x.min())

def rgb_to_lab(image):
    """
    Convert RGB image to LAB image.
    """
    chroma = lab.rgb_to_lab(image.clone())
    chroma[0] /= 100.
    chroma[1:] /= 127.
    chroma[1:] = (chroma[1:] + 1)/2
    return chroma

def exr_to_tensor(exrfile):
    """
    Load an exr file and return a tensor of the same shape.
    """
    im = np.flip(cv2.imread(exrfile, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), -1)
    return torch.tensor(im.copy()).permute(-1, 0, 1)

def correct_exposure(im):
    im = torch.log(1 + im)
    im /= 8.0 * im.mean()
    im = torch.exp(im) - 1.0
    im = torch.pow(im.clip(0, 1), 1 / 2.4)
    return im

def image_float_to_uint8(img):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    vmin = np.min(img)
    vmax = np.max(img)
    if vmax - vmin < 1e-10:
        vmax += 1e-10
    img = (img - vmin) / (vmax - vmin)
    img *= 255.0
    return img.astype(np.uint8)

def cmap(img, color_map=cv2.COLORMAP_HOT):
    """
    Apply 'HOT' color to a float image
    """
    # print("min/max:", img.min(), img.max())
    img = image_float_to_uint8(img)
    img_cmap = cv2.applyColorMap(img, color_map) #(img * 255).astype(np.uint8)
    img_cmap = np.flip(img_cmap, -1).transpose(2, 0, 1)
    # print("cmap:", img_cmap.min(), img_cmap.max())
    return img_cmap

def repeat_interleave(arr, n):
    """
    Repeat the elements of an array n times and interleave them.
    """
    shape = arr.shape
    return arr.reshape(shape[0], -1).unsqueeze(1).repeat(1, n, 1).view(n * shape[0], *shape[1:])

def apply_cmap_to_batch(similarities, rand_locations):
    B, H, W = similarities.shape
    results = []
    for i in range(B):
        h, w = rand_locations[i]
        sim_cmap = cmap(similarities[i])
        sim_cmap[:, h-3:h+3, w-3:w+3] = np.array([0, 0, 255])[:, None, None]
        results.append(sim_cmap)
    return np.stack(results)/255.

def interpolate_nan(grid):
    """ Given grid of shape (3, H, W), replace the values in the grid that are nan with interpolated values of the grid neighboring pixels"""
    grid = grid.permute(1, 2, 0).clone().numpy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]).any():
                grid[i, j] = np.nanmean(grid[max(0, i-1):min(grid.shape[0], i+2), max(0, j-1):min(grid.shape[1], j+2)], axis=(0, 1))
    return torch.tensor(grid).permute(2, 0, 1)

def normalize(x):
    return (x - x.min())/(x.max() - x.min())

def get_path_with_name(arr, name):
    return [f for f in arr if name in f][0]

def random_crop(image, mat_label, size):
    _, H, W = image.shape
    start_h, start_w = torch.randint(H - size[0], (1,))[0], torch.randint(W - size[1], (1,))[0]
    image = image[..., start_h:start_h + size[0], start_w:start_w + size[1]]
    mat_label = mat_label[..., start_h:start_h + size[0], start_w:start_w + size[1]]
    return image, mat_label

def top_left_crop(image, mat_label, size):
    _, H, W = image.shape
    image = image[..., :size[0], :size[1]]
    mat_label = mat_label[..., :size[0], :size[1]]
    return image, mat_label

def center_crop(image, mat_label, size):
    _, H, W = image.shape
    start_h, start_w = (H - size[0])//2, (W - size[1])//2
    image = image[..., start_h:start_h + size[0], start_w:start_w + size[1]]
    mat_label = mat_label[..., start_h:start_h + size[0], start_w:start_w + size[1]]
    return image, mat_label

def flip_lr(image, mat_label):
    """
    Flip the image and label horizontally.
    """
    if np.random.rand() < 0.5:
        image = torch.flip(image, (-1,))
        mat_label = torch.flip(mat_label, (-1,))

    return image, mat_label

def augment_data(image, mat_label, size, flip=True, test=False):
    """
    Augment data by cropping and flipping the image and label horizontally and vertically.
    """
    if test:
        image, mat_label = center_crop(image, mat_label, size)
    else:
        image, mat_label = random_crop(image, mat_label, size)
        if flip:
            image, mat_label = flip_lr(image, mat_label)
    return image, mat_label

def convert_labels_to_color(labels):
    """
    Given a batch of labels map of dimension [B, H, W], convert them to a color image with 
    one color per label image by giving each label a unique color."""
    color_maps = torch.zeros((labels.shape[0], labels.shape[1], labels.shape[2], 3))
    unique_labels = torch.unique(labels)
    for i, label in enumerate(unique_labels):
        if label == 0:
            color_maps[labels == label, :] = torch.tensor([0.7, 0.7, 0.7])
        else:
            color_maps[labels == label, :] = torch.rand(3)
    return color_maps.permute(0, 3, 1, 2)

def MIOU_score(pred, target):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / union


def get_combined_visualization(viz_data):
    """
    viz_data: A dictioning containing the following keys: images, mat_labels, embeddings
    """
    images = viz_data["images"]
    mat_labels = viz_data["mat_labels"]
    embeddings = viz_data["embeddings"]
    B, C, H, W = images.shape

    rand_h, rand_w = np.random.randint(0, H), np.random.randint(0, W)
    similarities, heatmaps, binary_masks, mask_binary_diff = contrastive_loss.embedding_similarity(embeddings, mat_labels, rand_h, rand_w)
    mask_pred = similarities.unsqueeze(1).repeat(1, 3, 1, 1)
    mask_gt = binary_masks

    image_grid_np = (torchvision.utils.make_grid(
        images.cpu(), nrow=B).numpy() + 1) / 2

    heatmap_grid_np = torchvision.utils.make_grid(
        heatmaps.cpu(), nrow=B).numpy()

    mat_labels_colored = convert_labels_to_color(mat_labels.cpu())
    mat_labels_colored[..., rand_h-2:rand_h+2, rand_w-2:rand_w + 2] = torch.tensor([1., 0., 0.])[None, :, None, None]
    mat_labels_colored_np = torchvision.utils.make_grid(mat_labels_colored.cpu(), nrow=B).numpy()

    mask_pred_grid_np = torchvision.utils.make_grid(
        mask_pred.cpu(), nrow=B).numpy()
    mask_gt_grid_np = torchvision.utils.make_grid(
        mask_gt.cpu(), nrow=B).numpy()

    mask_binary_diff_grid_np = torchvision.utils.make_grid(
        mask_binary_diff.cpu(), nrow=B).numpy()

    visual_grids = [image_grid_np]    
    visual_grids.append(mat_labels_colored_np)
    visualization = np.hstack(visual_grids + 
                                [heatmap_grid_np,
                                mask_pred_grid_np,
                                (mask_pred_grid_np > 0.2).astype(float),
                                (mask_pred_grid_np > 0.4).astype(float),
                                (mask_pred_grid_np > 0.5).astype(float),
                                (mask_pred_grid_np > 0.6).astype(float),
                                (mask_pred_grid_np > 0.8).astype(float),
                                mask_gt_grid_np,
                                mask_binary_diff_grid_np])
    return visualization

def get_classification_visualization(viz_data):
    images = viz_data["images"]
    B, C, H, W = images.shape
    mat_labels = viz_data["mat_labels"]
    scores = viz_data["scores"]
    path1 = viz_data["path1"]
    context_embeddings_1 = viz_data["context_embeddings_1"]
    sz = int((viz_data["layer_1"].shape[1] - 1) ** 0.5)
    layer_1 = viz_data["layer_1"][:, 1:].reshape(B, sz, sz, 768).permute(0, 3, 1, 2)
    
    B, C, H, W = images.shape
    reference_location = viz_data["reference_locations"]
    mask_pred = (scores > 0.5).float().unsqueeze(1).repeat(1, 3, 1, 1)
    for i in range(B):
        start_h, start_w = reference_location[i, 0] - 5, reference_location[i, 1] - 5
        # unnormalize the image using the mean and std
        images[i] = images[i] * torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(images.device) + torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(images.device)
        images[i, :, start_h:start_h+5, start_w:start_w+5] = torch.tensor([1., 0., 0.])[:, None, None]
    
    image_grid_np = torchvision.utils.make_grid(images.cpu(), nrow=B).numpy()
    mat_labels = mat_labels.unsqueeze(1).repeat(1, 3, 1, 1)
    mat_labels_np = torchvision.utils.make_grid(mat_labels.cpu(), nrow=B).numpy()
    
    # turn mask tensor into a grid
    mask_pred_grid_np = torchvision.utils.make_grid(mask_pred.cpu(), nrow=B).numpy()
    scores_grid_np = torchvision.utils.make_grid(scores.detach().cpu().unsqueeze(1).repeat(1, 3, 1, 1) , nrow=B).numpy()
    path1_viz = torchvision.utils.make_grid(path1[:, :3].detach().cpu(), nrow=B).numpy()
    context_embeddings_1_viz = torchvision.utils.make_grid(context_embeddings_1[:, :3].detach().cpu(), nrow=B).numpy()
    layer_1_viz = torchvision.utils.make_grid(layer_1[:, :3].detach().cpu(), nrow=B).numpy()
    visual_grids = [image_grid_np, mat_labels_np, mask_pred_grid_np, scores_grid_np]    
    visualization = np.hstack(visual_grids)
    return visualization, path1_viz, None, None, None, context_embeddings_1_viz, None, None, None, layer_1_viz, None, None, None

def get_optimal_threshold(similarities, mask_gt):
    """
    Given a tensor of similarities, find the optimal threshold that maximizes the MIOU score
    """
    similarities = similarities.flatten()
    mask_gt = mask_gt.flatten()
    all_mious = []
    best_threshold = 0
    best_score = 0
    for threshold in np.linspace(0, 1.01, 1000):
        binary_mask = (similarities > threshold)
        score = MIOU_score(binary_mask, mask_gt)
        all_mious.append(score)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, all_mious

def get_test_transformer_visualizations(viz_data):
    """
    viz_data: A dictioning containing the following keys: images, mat_labels, embeddings
    """
    images = viz_data["images"]
    mat_labels = viz_data["mat_labels"]
    scores = viz_data["scores"]
    reference_locations = viz_data["reference_locations"]
    image = images[0] * torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(images.device) + torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(images.device)
    ref_h, ref_w =  reference_locations[0, 0], reference_locations[0, 1]
    image_np = image.cpu().numpy()
    
    optimal_threshold, all_mious = get_optimal_threshold(scores[0].detach().cpu().numpy(), mat_labels[0].detach().cpu().numpy())
    optimal_predicted_mask = (scores[0].detach() > optimal_threshold).float().cpu().unsqueeze(0).repeat(3, 1, 1).numpy()
    
    scores_np = scores[0].detach().unsqueeze(0).repeat(3, 1, 1).cpu().numpy()
    
    visualizations = {"image": image_np,
                      "binary_mask": mat_labels[0].unsqueeze(0).repeat(3, 1, 1).cpu().numpy(),
                     "optimal_predicted_mask": optimal_predicted_mask,
                      "optimal_threshold": optimal_threshold,
                      "scores": scores_np,
                      "all_mious": all_mious}
    return visualizations

def convert_to_opencv_image(im):
    im = im.transpose(1, 2, 0)
    im = (im * 255).astype(np.uint8)
    return np.flip(im, axis=-1)
