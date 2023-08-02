import os
import cv2
import gradio as gr
import numpy as np 
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage import color as color_module
import src.models.model_utils_pl as model_utils
import src.utils.args_test as parse_args
import src.utils.utils as utils
from PIL import Image

args, conf = parse_args.parse_args()

def overlay_image(img, mask, color, alpha=0.8):
    orig_mask = mask.copy()
    mask = mask.copy().transpose(1, 2, 0).astype(np.float32) 
    img = img.transpose(1, 2, 0)
    masked_image_gt = np.flip(img, axis=-1)
    mask[..., 0] *= color[0]
    mask[..., 1] *= color[1]
    mask[..., 2] *= color[2]
    img_hsv = color_module.rgb2hsv(masked_image_gt)
    color_mask_hsv = color_module.rgb2hsv(mask)

    idx = np.where(orig_mask[0] > 0.)
    img_hsv[idx[0], idx[1], 0] = color_mask_hsv[idx[0], idx[1], 0]
    img_hsv[idx[0], idx[1], 1] = color_mask_hsv[idx[0], idx[1], 1]  * alpha
    img_masked = (color_module.hsv2rgb(img_hsv) * 255).astype(np.uint8)
    return np.flip(img_masked, -1)

def inference(img, reference_locations, thresh, image_orig, session_state):
    with torch.no_grad():
        scores, path1, path2, path3, path4, context_embeddings_1, context_embeddings_2, context_embeddings_3, context_embeddings_4, layer_1, layer_2, layer_3, layer_4 = net.net(img, reference_locations)

    session_state["scores"] = scores
    predicted_mask = (scores[0].detach() > thresh).float().cpu().unsqueeze(0).repeat(3, 1, 1).numpy()
    predicted_overlayed_image = overlay_image(image_orig, predicted_mask, (0, 0, 255))
    return predicted_mask.transpose(1, 2, 0), predicted_overlayed_image, session_state

def get_select_coords(img, thresh, session_state, evt: gr.SelectData):

    h, w = evt.index[1], evt.index[0]
    out = session_state["image_orig"].copy()
    out = out.astype(np.uint8)
    sq_sz = 7
    out = cv2.rectangle(out, (w - sq_sz, h - sq_sz), (w + sq_sz, h + sq_sz), (255, 0, 0), 2)
    
    predicted_mask, predicted_overlayed_image, session_state = inference(session_state["image_torch"].unsqueeze(0).cuda(), 
                                                          torch.tensor([[h, w]]), 
                                                          session_state["threshold"], 
                                                          session_state["image_orig"].copy().transpose(-1, 0, 1), 
                                                          session_state)
    session_state["predicted_mask"] = predicted_mask
    session_state["predicted_overlayed_image"] = predicted_overlayed_image
    
    return out, predicted_overlayed_image, session_state

def preprocess_size(img, session_state):
    # Resize so that the smaller dimension is 512, and then take center crop
    image_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image_pil = Image.fromarray(img)
    H, W = image_pil.size
    sz = 512
    if H < W:
        image = image_pil.resize((sz, int(sz * W / H)), Image.BOX)
    else:
        image = image_pil.resize((int(sz * H / W), sz), Image.BOX)
    image = transforms.ToTensor()(image)
    image = transforms.CenterCrop((512, 512))(image)
    session_state["image_orig"] = (image.permute(1, 2, 0).numpy().copy() * 255).astype(np.uint8)
    session_state["image_torch"] = image_transform(image)
    
    return session_state["image_orig"].copy(), session_state

def apply_threshold(thresh, session_state):
    session_state["threshold"] = thresh    
    if session_state["image_orig"] is None:
        return None
        
    if session_state["scores"] is not None:
        predicted_mask = (session_state["scores"][0].detach() > thresh).float().cpu().unsqueeze(0).repeat(3, 1, 1).numpy()
        predicted_overlayed_image = overlay_image(session_state["image_orig"].transpose(-1, 0, 1), predicted_mask, (0, 0, 255))
        predicted_mask = predicted_mask.transpose(1, 2, 0)
    session_state["predicted_mask"] = predicted_mask
    session_state["predicted_overlayed_image"] = predicted_overlayed_image
    return predicted_overlayed_image, session_state

def clear_all(img, session_state):
    session_state["image_orig"] = None
    session_state["image_torch"] = None
    session_state["scores"] = None
    session_state["predicted_mask"] = None
    session_state["predicted_overlayed_image"] = None
    return None, None, session_state

def generate_overlay(session_state):
    return session_state["predicted_overlayed_image"]
    
def generate_scores(session_state):
    if session_state["scores"] is None:
        return None
    return (session_state["scores"].detach().cpu().repeat(3, 1, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    
def generate_mask(session_state):
    if session_state["predicted_mask"] is None:
        return None
    return (session_state["predicted_mask"] * 255).astype(np.uint8)

TITLE = '# [Materialistic: Selecting Similar Materials in Images](https://prafullsharma.net/materialistic)'

USAGE = '''To run the demo, you should:   
    1. Upload your image. Note that the current demo resizes the image and center crops the image to only run on 512x512 images.
    2. **Click on a given pixel of the image**   
    3. Adjust the threshold to finetune the mask. Higher threshold means removing pixels with small similarity scores from the predicted mask.

    4. You can view the prediction overlaid over the image, binary mask, and the per-pixel scores using the buttons.
    '''

with gr.Blocks() as app:

    gr.Markdown(TITLE)
    gr.Markdown(USAGE)
    session_state = gr.State({"image_orig": None,
                              "image_torch": None,
                              "scores": None,
                              "predicted_mask": None,
                              "predicted_overlayed_image": None,
                              "threshold": 0.5})
    
    net = model_utils.create_model(conf, args)
    net = net.cuda()
    net.load_checkpoint(args.checkpoint_dir)
    net.eval()

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input")
            threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="Threshold", value=0.5, interactive=True)
            overlay_button = gr.Button('Overlay Result', size="sm")
            mask_button = gr.Button('Binary Mask', size="sm")
            scores_button = gr.Button('Per-pixel Scores', size="sm")
            
        with gr.Column(scale=1):
            output_image = gr.Image(label="Output")
            
            
    input_image.upload(preprocess_size, [input_image, session_state], [input_image, session_state])
    input_image.select(get_select_coords, [input_image, threshold, session_state], [input_image, output_image, session_state])
    input_image.clear(clear_all, [input_image, session_state], [input_image, output_image, session_state])
    threshold.change(apply_threshold, [threshold, session_state], [output_image, session_state])
    overlay_button.click(generate_overlay, session_state, output_image)
    mask_button.click(generate_mask, session_state, output_image)
    scores_button.click(generate_scores, session_state, output_image)
    
app.queue(api_open=False, concurrency_count=1)
app.launch(share=False,
            debug=False,
            server_name="0.0.0.0",
           ssl_verify=False)
