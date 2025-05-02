import torch
import os
import numpy as np
from collections import defaultdict

from third_party import (
    parse_arguments,
    get_model, 
    load_checkpoint, 
    prepare_data, 
    extract_study_id,
    get_target_layer,
    generate_gradcam_heatmap,
    process_heatmap,
    overlay_heatmap_on_img,
    visualize_heatmap,
    save_heatmap,
    class_distinctiveness,
    sum_up_distinctiveness,
    #normalize_distinctiveness, # Called in plot_distinctiveness_boxplots if normalize=True
    plot_distinctiveness_boxplots)

# To access all the default arguments from run_models.pys
args = parse_arguments()

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
        'Pleaural Effusion':10
        }

    # Storage locations
    map_dir = os.path.expanduser('~/repo/xai_thesis/saliency_maps/')
    os.makedirs(map_dir, exist_ok=True)

    cache_dir = os.path.expanduser('~/repo/xai_thesis/heatmap_cache')
    man_path = 'ckpt_d_ignore_3'
    normalize = True
    cache_dir = os.path.join(cache_dir, args.model, man_path)
    os.makedirs(cache_dir, exist_ok=True)


    # Load model, img data, and access layer to generate saliency maps
    model = load_model(model_name=args.model, tasks=tasks, model_args=args)
    layer = get_target_layer(model=model)
    data_loader = prepare_data(args)
    saliency_dict = defaultdict(list)
    distinctiveness_collection = {}

    # Access the patient IDs when storing saliency maps (heatmap overlay with original iamge)
    df = extract_study_id(mode=args.run_test)
    ids = df['study_id'].str.split('/', expand=True)[1] 

    # Loop over img data
    for i, (img, _) in enumerate(data_loader):
        id = ids[i]

        # Loop over tasks
        for target_name, idx in target_class_dict.items():
            cache_map_path = os.path.join(cache_dir, f"{id}_{target_name}.npz")

            # If already existing load heatmaps otherwise compute them
            if args.saliency=='get':
                # Load pre-computed heatmaps
                with np.load(cache_map_path) as dict_lookup:
                    heatmap = dict_lookup['heatmap']
            elif args.saliency=='compute':
                heatmap = generate_gradcam_heatmap(model=model, 
                                                input_tensor=img, 
                                                target_class=idx, 
                                                target_layer=layer)
                np.savez_compressed(cache_map_path, heatmap=heatmap)  

            # If images shall be computed first check if already existing, otherwise compute.
            elif args.saliency=='save_img':
                if os.path.exists(cache_map_path):
                    with np.load(cache_map_path) as d:
                        heatmap = d['heatmap']
                else:
                    heatmap = generate_gradcam_heatmap(
                        model=model,
                        input_tensor=img,
                        target_class=idx,
                        target_layer=layer
                    )
                    np.savez_compressed(cache_map_path, heatmap=heatmap)
                try:
                    # Process and save GradCam overlay to original img.
                    upscaled_heatmap = process_heatmap(heatmap=heatmap, target_size=(320, 320))
                    overlayed_imgs = overlay_heatmap_on_img(original_img=img, heatmap=upscaled_heatmap, alpha=0.4)
                    fig = visualize_heatmap(heatmap=overlayed_imgs, title=(f"Grad-CM for {id} on {target_name}"))
                    save_path = os.path.join(map_dir, f"{args.model}/{target_name}_{id}.png")
                    #print("Save path:", save_path)
                    save_heatmap(fig, save_path)
                except LookupError:
                    print("Heatmap not existing. Make sure it is available in correct location.")
            else:
                raise ValueError("Unknown saliency mode.")

            heatmap_vector = process_heatmap(heatmap=heatmap, 
                                      target_size=(10, 10), 
                                      normalize=True, 
                                      flatten=True,
                                      as_tensor=False)
            
            # Every round append the heatmap_vector to the corresponding target_name (phathology)
            saliency_dict[target_name].append(heatmap_vector)
        
        distinctiveness = class_distinctiveness(saliency_dict)
        distinctiveness_collection = sum_up_distinctiveness(distinctiveness_collection, distinctiveness)
    
    save_dir = os.path.expanduser(f"~/repo/xai_thesis/distinctiveness/{args.model}")
    save_path = os.path.join(save_dir, "distinctiveness_boxplot.png")
    fig = plot_distinctiveness_boxplots(distinctiveness_collection,
                                        normalize=normalize,
                                        save_path=save_path)


    # print("Pairwise cosine‐similarities (distinctiveness):")
    # for (c1, c2), score in distinctiveness_collection.items():
    #     print(f"  {c1:>7} vs {c2:<7} → {score}")
               
    print("********** Finished saliency_maps script **********")   
    print("Added this test for merge into main.")


if __name__ == "__main__":
    main()



