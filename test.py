import os
import torch
import numpy as np
import cv2
import pickle
import src.loss_functions.loss_utils as loss_utils
import src.models.model_utils_pl as model_utils
import src.data.real_dataset as real_dataset
import src.utils.utils as utils
import src.utils.args_test as parse_args
from skimage import color as color_module

torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
np.random.seed(0)

args, conf = parse_args.parse_args()

print("Using device: ", torch.cuda.get_device_name(0))

def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)


BATCH_SIZE = 1
H, W = args.image_size, args.image_size

using_transformer = True if "transformer" in conf.model.model_type else False
test_dataset = real_dataset.RealImageDataset("../data/combined_real_test_data/", 
                                            use_chroma=args.use_chroma, 
                                            use_depth=args.use_depth, 
                                            use_normal=args.use_normal, 
                                            size=(H, W), test=True, using_transformer=using_transformer)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=1, 
                                            shuffle=False, 
                                            num_workers=1, 
                                            pin_memory=True, 
                                            worker_init_fn=worker_init_fn)


###### Model ######
# create model
net = model_utils.create_model(conf, args)
net = net.cuda()
net.load_checkpoint(args.checkpoint_dir)
net.eval()


def overlay_image(img, mask, color, alpha=0.6):
    orig_mask = mask.copy()
    mask = mask.copy().transpose(1, 2, 0).astype(np.float32) 
    masked_image_gt = utils.convert_to_opencv_image(img).copy().astype(np.float32)/255.
    color = (41, 215, 162)
    mask[..., 0] *= color[0]
    mask[..., 1] *= color[1]
    mask[..., 2] *= color[2]
    img_hsv = color_module.rgb2hsv(masked_image_gt)
    color_mask_hsv = color_module.rgb2hsv(mask)

    idx = np.where(orig_mask[0] > 0.)
    img_hsv[idx[0], idx[1], 0] = color_mask_hsv[idx[0], idx[1], 0]
    img_hsv[idx[0], idx[1], 1] = color_mask_hsv[idx[0], idx[1], 1]  * alpha
    img_masked = (color_module.hsv2rgb(img_hsv) * 255).astype(np.uint8)
    return img_masked

criterion = loss_utils.create_loss_function(conf)
miou_scores = []
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)
    
for i, data in enumerate(test_loader):
    image = data['images'].cuda()
    sample_path = data['sample_path']
    input = image.cuda()
    
    mat_labels = data["mat_labels"].cuda()
    reference_locations = data["reference_locations"].cuda()
    if os.path.exists(os.path.join(args.results_dir, "reference_locations", f"{i}.pkl")):
        reference_locations = pickle.load(open(os.path.join(args.results_dir, "reference_locations", f"{i}.pkl"), "rb"))["reference_locations"]
    mat_labels = (mat_labels == mat_labels[0, reference_locations[0, 0], reference_locations[0, 1]]).float()

    with torch.no_grad():
        scores, path1, path2, path3, path4, context_embeddings_1, context_embeddings_2, context_embeddings_3, context_embeddings_4, layer_1, layer_2, layer_3, layer_4 = net.net(input, reference_locations)
    viz_data = {"images": image,
                    "scores": scores,
                    "mat_labels": mat_labels,
                    "reference_locations": reference_locations}

    visualizations = utils.get_test_transformer_visualizations(viz_data)

    # image, material_labels, optimal_predicted_mask, optimal threshold, binary_mask
    image = visualizations['image']
    binary_mask = visualizations['binary_mask']

    optimal_predicted_mask = visualizations['optimal_predicted_mask']
    optimal_threshold = visualizations['optimal_threshold']
    scores = visualizations['scores']
    all_mious = visualizations['all_mious']
    miou_score = utils.MIOU_score(binary_mask[0], optimal_predicted_mask[0])
    miou_scores.append(miou_score)
    print("\nmiou score: ", np.around(miou_score, 3))
    print("Optimal threshold: ", np.around(optimal_threshold, 3))
    
    if not os.path.isdir(os.path.join(args.results_dir, "input_rgb")):
        os.mkdir(os.path.join(args.results_dir, "input_rgb"))
    cv2.imwrite(os.path.join(args.results_dir, "input_rgb", f"{i}.png"), utils.convert_to_opencv_image(image))

    if not os.path.isdir(os.path.join(args.results_dir, "all_mious" + args.method_name)):
        os.mkdir(os.path.join(args.results_dir, "all_mious" + args.method_name))
        
    pickle.dump(all_mious, open(os.path.join(args.results_dir, "all_mious" + args.method_name, f"{i}.pkl"), "wb"))
   
    if not os.path.isdir(os.path.join(args.results_dir, "scores_" + args.method_name)):
            os.mkdir(os.path.join(args.results_dir, "scores_" + args.method_name))
    cv2.imwrite(os.path.join(args.results_dir, "scores_" + args.method_name, f"{i}.png"), utils.convert_to_opencv_image(scores))

    img = data['images']
    img = img[0] * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    img = img.cpu().numpy()
    annotated_image = utils.convert_to_opencv_image(img).copy()
    viz_x = reference_locations[0, 0].item()
    viz_y = reference_locations[0, 1].item()
    sq_sz = 7
    annotated_image = cv2.rectangle(annotated_image, (viz_y - sq_sz, viz_x - sq_sz), (viz_y + sq_sz, viz_x + sq_sz), (41, 215, 162), 2)
    
    masked_image_gt = overlay_image(img, binary_mask, (0, 0, 255))
        
    masked_image_pred = overlay_image(img, optimal_predicted_mask, (0, 0, 255))
    
    if not os.path.isdir(os.path.join(args.results_dir, args.method_name)):
        os.mkdir(os.path.join(args.results_dir, args.method_name))
        
    # save optimal predicted mask to result/
    cv2.imwrite(os.path.join(args.results_dir, args.method_name, f"{i}.png"), masked_image_pred)

    if not os.path.isdir(os.path.join(args.results_dir, "gt_masks")):
        os.mkdir(os.path.join(args.results_dir, "gt_masks"))
    # save binary mask to result/gt_masks
    cv2.imwrite(os.path.join(args.results_dir, "gt_masks", f"{i}.png"), masked_image_gt)
    
    scores_np = utils.convert_to_opencv_image(scores).copy()
    
    # vertically stack the images with 5 pixel difference between them
    white_space = np.ones((5, annotated_image.shape[1], 3), dtype=np.uint8) * 255
    stacked_image = np.vstack((annotated_image, white_space, masked_image_gt, white_space, masked_image_pred, white_space, scores_np))
    
    if not os.path.isdir(os.path.join(args.results_dir, "stacked_images_"+args.method_name)):
        os.mkdir(os.path.join(args.results_dir, "stacked_images_"+args.method_name))
    # save stacked image to result/stacked_images
    cv2.imwrite(os.path.join(args.results_dir, "stacked_images_"+args.method_name, f"{i}.png"), stacked_image)
    
    # save optimal threshold to result/method_name
    with open(os.path.join(args.results_dir, args.method_name, "optimal_threshold_" + str(i) + "+.txt"), "a") as f:
        f.write(f"{optimal_threshold}\n")

print("\n\n\nMIOU: ", np.mean(miou_scores))
with open(os.path.join(args.results_dir, args.method_name, "miou_scores.pkl"), "wb") as f:
    pickle.dump(miou_scores, f)
