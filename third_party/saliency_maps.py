import os
import numpy as np
import datetime
import torch
import json
import logging
from collections import defaultdict
from tqdm import tqdm

import utils
from run_models import parse_arguments, get_model, load_checkpoint, prepare_data, DEVICE


logging.basicConfig(level=logging.DEBUG)  # or DEBUG for more verbosity
logger = logging.getLogger(__name__)


local_time_zone = datetime.timezone(datetime.timedelta(hours=2), name="CEST")
start = datetime.datetime.now(local_time_zone)

# Access all the default arguments from run_models.pys
args = parse_arguments()

"""
This script is to generate saliency maps. In the run_models.py script set the arument 'saliency'
Whether to compute and save="compute", retreive stored="get", or compute and save imgage_maps="save_img"
batch_size = 1, otherwise it doesen't go correctly through tensor.
"""
args.batch_size = 1

# Assume config.json sits in the same folder as this script.
# If you put it elsewhere, just change the path below.
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "/home/fkirchhofer/repo/xai_thesis/config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
_ctx = utils._flatten_context(config)

# 2) Expand placeholders everywhere they appear
config = utils._expand_placeholders(config, _ctx)

# 3) Normalize saliency paths: expand ~ and join base_dir with folder names
config = utils._expand_and_join_base_dirs(config)
saliency_cfg = config.get("saliency_script", {})

# Helper to expand "~" in all paths
def expand(p):
    return os.path.expanduser(p)
# Base directory
base_dir = expand(saliency_cfg.get("base_dir", "~/repo/xai_thesis"))

# SALIENCY sub‐section:
sal_cfg = saliency_cfg.get("saliency", {})
method = sal_cfg.get("method", "gradcam").lower()
map_folder       = sal_cfg.get("map_folder", "saliency_maps")
cache_folder     = sal_cfg.get("cache_folder", "heatmap_cache")
manual_folder_name    = sal_cfg.get("manifold_name", "ckpt_i_ignore_1")


eval_cfg = config.get("evaluation")
if "evaluate_test_set" in eval_cfg:
    args.run_test = utils._coerce_bool(eval_cfg["evaluate_test_set"])
    print(f"[INFO] Overriding args_model.run_test -> {args.run_test} "
        f"(from config['evaluation']['evaluate_test_set'])")

baseline_tensor = None
if method in ("ig", "deeplift"):
    try:
        saliency_baseline_path = "/home/fkirchhofer/repo/xai_thesis/third_party/mean_image.pt"
        baseline_tensor = torch.load(saliency_baseline_path)
        if baseline_tensor.ndim == 3:
            baseline_tensor = baseline_tensor.unsqueeze(0)
        baseline_tensor = baseline_tensor.to(DEVICE)
    except FileNotFoundError:
        logger.warning(f"Baseline file not found: {saliency_baseline_path}.")

# DISTINCCTIVENESS sub‐section:
dist_cfg = saliency_cfg.get("distinctiveness", {})
dist_func        = dist_cfg.get("function", "cosine_similarity")
dist_output_root = dist_cfg.get("output_folder", "distinctiveness_cosine_similarity")

def load_model(model_name:str='DenseNet121', tasks:list=None, model_args=None)->torch.nn.Module:
    """
    Load an initialize trained model in eval mode.
    Args:
        model_name (str): Either 'DenseNet121', 'ResNet152', 'Inceptionv4'. By default 'DenseNet121'.
        tasks (list): List with pathologies the model will look for and classify for its presence.
        model_args (various): See run_models.py
    Return:
        Initialized model in evaluation mode (torch.nn.Module).
    """
    # Use parsed arguments as default if none provided
    if model_args is None:
        model_args = args
    
    
    model = get_model(model_name, tasks, model_args)
    model = load_checkpoint(model, model_args.ckpt)
    return model


def main():
    """
    Generate (or fetch) saliency maps and (optionally) distinctiveness JSONs.

    Workflow
    --------
    1) Parse saliency config (method, folders, baseline for IG/DeepLIFT).
    2) Load model & target layer, prepare DataLoader (val or test).
    3) For each sample and target class (5 CheXpert eval classes), either:
       - 'compute': compute heatmap and cache (*.npz),
       - 'get'    : load heatmap from cache (fallback: compute),
       - 'save_img': compute/load and save overlay images.
    4) (Optional) Aggregate per-class heatmaps into a class-wise distinctiveness
       measure and write per-model JSON.

    Side Effects
    ------------
    - Writes heatmaps (*.npz) under cache folder.
    - Optionally writes PNG overlays.
    - Optionally writes class-wise distinctiveness JSONs per model.
    """
    print("\n************************ Get started with saliency maps script ************************")
    
    # Some general definitions
    tasks = [
    'No Finding', 'Enlarged Cardiomediastinum' ,'Cardiomegaly', 
    'Lung Opacity', 'Lung Lesion' , 'Edema' ,
    'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]
    target_class_dict = {
        'Cardiomegaly':2,
        'Edema':5,
        'Consolidation':6,
        'Atelectasis':8,
        'Pleural Effusion':10 # Typo fix: Pleural Effusion
        }

    # Map directory to store overlay images
    map_dir = os.path.join(base_dir, map_folder)
    os.makedirs(map_dir, exist_ok=True)

    # Directory to cache the heatmaps
    cache_root = os.path.join(base_dir, cache_folder)
    cache_dir  = os.path.join(cache_root, args.model, manual_folder_name)
    os.makedirs(cache_dir, exist_ok=True)

    # Load model, img data, and access layer to generate saliency maps
    model = load_model(model_name=args.model, tasks=tasks, model_args=args)
    model = model.eval().to(DEVICE)
    layer = utils.get_target_layer(model=model, method=method).to(DEVICE)
    logger.info(f"Saliency method: {method}\nModel: {model.__class__.__name__}\nPassed layer: {layer}")
    data_loader = prepare_data(args)
    distinctiveness_collection = {}

    # Access the patient IDs when storing saliency maps (heatmap overlay with original iamge)
    df = utils.extract_study_id(mode=args.run_test)
    ids = df['study_id'].str.split('/', expand=True)[1] 
    
    #torch.cuda.empty_cache()

    # Loop over img data
    for i, (img, _) in enumerate(tqdm(data_loader, desc="Processing batches")):
        saliency_dict = defaultdict(list)
        # _ is the label tensor (which we don't need)
        study_id = ids[i]
        img = img.to(DEVICE, non_blocking=True)

        # Loop over tasks
        for target_name, idx in target_class_dict.items():
            cache_map_path = os.path.join(cache_dir, f"{study_id}_{target_name}.npz")
            # If already existing load heatmaps otherwise compute them
            if args.saliency=='get':
                with np.load(cache_map_path) as dict_lookup:
                    heatmap = dict_lookup['heatmap']
            elif args.saliency=='compute':
                    if method == "gradcam":
                        heatmap = utils.generate_gradcam_heatmap(
                            model=model,
                            input_tensor=img,
                            target_class=idx,
                            target_layer=layer)
                        np.savez_compressed(cache_map_path, heatmap=heatmap)  
                    elif method == "lrp":
                        heatmap = utils.generate_lrp_attribution(
                            model=model,
                            input_tensor=img,
                            target_class=idx,
                            target_layer=layer)
                        np.savez_compressed(cache_map_path, heatmap=heatmap)                         
                    elif method == 'ig':
                        heatmap = utils.ig_heatmap(
                            model=model,
                            input_tensor=img,
                            target_class=idx,
                            target_layer=layer,
                            baseline=None) # Switch to None to set zero_baseline or baseline_tensor
                        np.savez_compressed(cache_map_path, heatmap=heatmap)
                    elif method == "deeplift":
                        heatmap = utils.deep_lift_layer_heatmap(
                            model=model,
                            layer=layer,
                            input_tensor=img,
                            target_class=idx,
                            baseline=None) # Switch to None to set zero_baseline or baseline_tensor
                        np.savez_compressed(cache_map_path, heatmap=heatmap)  
                    else:
                        raise ValueError(f"Unknown saliency method: {method}")
                    

            # If images shall be computed first check if already existing, otherwise compute.
            elif args.saliency=='save_img':
                heatmap = None
                if os.path.exists(cache_map_path):                    
                    try:
                        with np.load(cache_map_path) as d:
                            heatmap = d['heatmap']
                    except LookupError:
                        print("Heatmap not existing. Make sure it is available.")
                if heatmap is None:
                    if method == "gradcam":
                        heatmap = utils.generate_gradcam_heatmap(
                            model=model,
                            input_tensor=img,
                            target_class=idx,
                            target_layer=layer)
                        np.savez_compressed(cache_map_path, heatmap=heatmap)  
                    elif method == "lrp":
                        heatmap = utils.generate_lrp_attribution(
                            model=model,
                            input_tensor=img,
                            target_class=idx,
                            target_layer=layer)
                        np.savez_compressed(cache_map_path, heatmap=heatmap) 
                    elif method == 'ig':
                        heatmap = utils.ig_heatmap(
                            model=model,
                            input_tensor=img,
                            target_class=idx,
                            target_layer=layer,
                            baseline=None) # Switch to None to set zero_baseline or baseline_tensor
                        np.savez_compressed(cache_map_path, heatmap=heatmap)
                    elif method == "deeplift":
                        heatmap = utils.deep_lift_layer_heatmap(
                            model=model,
                            layer=layer,
                            input_tensor=img,
                            target_class=idx,
                            baseline=None) # Switch to None to set zero_baseline or baseline_tensor
                        np.savez_compressed(cache_map_path, heatmap=heatmap)   
                    else:
                        raise ValueError(f"Unknown saliency method: {method}")
                try:
                    # Process and save heatmap overlay to original img.
                    processed_heatmap = utils.process_heatmap(heatmap=heatmap, saliency_method=method, target_size=(320, 320), as_tensor=False)
                    overlayed_imgs = utils.overlay_heatmap_on_img(original_img=img, heatmap=processed_heatmap, alpha=0.4)

                    method_title = sal_cfg.get("method", "gradcam")
                    title = (f"{method_title} for {study_id} on {target_name}")
                    fig = utils.visualize_heatmap(heatmap=overlayed_imgs, title=title)

                    save_model_folder = os.path.join(map_dir, args.model)
                    os.makedirs(save_model_folder, exist_ok=True)
                    save_path = os.path.join(save_model_folder, f"{target_name}_{study_id}.png")
                    utils.save_heatmap(fig, save_path)
                except LookupError:
                    print("Heatmap not existing. Make sure it is available in correct location.")
            else:
                raise ValueError(f"Unknown saliency map processing mode: {args.saliency}")
            heatmap_scaling = sal_cfg.get("heatmap_scaling") #heatmap.shape # 320,320 or 640,640 or 1280,1280 or heatmap.shape
            if heatmap_scaling == "original":
                heatmap_scaling = heatmap.shape
                heatmap_vector = utils.process_heatmap(heatmap=heatmap, 
                        target_size=heatmap_scaling, 
                        saliency_method=method, 
                        flatten=True)
            else:
                heatmap_vector = utils.process_heatmap(heatmap=heatmap, 
                                    target_size=((int(heatmap_scaling),int(heatmap_scaling))), 
                                    saliency_method=method, 
                                    flatten=True)
            
            # Every round append the heatmap_vector to the corresponding target_name (phathology)
            saliency_dict[target_name].append(heatmap_vector)
        
        distinctiveness = utils.class_distinctiveness(saliency_dict, function=dist_func)
        distinctiveness_collection = utils.sum_up_distinctiveness(
                                            collection=distinctiveness_collection,
                                            new_scores=distinctiveness)
    print(f"CHECK - {heatmap_scaling} heatmap scaling used for distinctiveness computation")
    save_dir = os.path.join(base_dir, dist_output_root, args.model)
    os.makedirs(save_dir, exist_ok=True)

    boxplot_path = os.path.join(save_dir, "distinctiveness_boxplot.png")
    utils.plot_distinctiveness_boxplots(distinctiveness_collection,
                                        normalize=True,
                                        save_path=boxplot_path)
    
    utils.per_class_distinctiveness(distinctiveness_collection=distinctiveness_collection,
                                  normalize=True,
                                  save_path=save_dir,
                                  ckpt_name=manual_folder_name)

    # Time measurement
    end = datetime.datetime.now(local_time_zone)
    delta = end-start
    total_seconds = int(delta.total_seconds())
    hours   = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    print(f"Elapsed time: {hours}h {minutes}m {seconds}s")

    print("************************ Finished saliency_maps script ************************")


if __name__ == "__main__":
    main()



