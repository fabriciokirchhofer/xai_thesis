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
    compute_centroids,
    compute_distinctiveness,
    plot_distinctiveness,
    get_target_layer,
    generate_gradcam_heatmap,
    process_heatmap,
    overlay_heatmap_on_img,
    visualize_heatmap,
    save_heatmap)

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device for XAI:", device)

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
    
    # Define output directory and ensure it exists.
    output_dir = os.path.expanduser('~/repo/xai_thesis/saliency_maps/')
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_name=args.model, tasks=tasks, model_args=args)
    layer = get_target_layer(model=model)

    if args.save_saliency:
        df = extract_study_id(mode=args.run_test)
        ids = df['study_id'].str.split('/', expand=True)[1] 
        
    data_loader = prepare_data(args)
    saliency_dict = defaultdict(list)


    for i, (img, _) in enumerate(data_loader):
        #img = img.to(device)
        if args.save_saliency:
            try:
                id = ids[i]
            except Warning:
                print(f"No saliency maps to be stored")

        for target_name, idx in target_class_dict.items():
            heatmap = generate_gradcam_heatmap(model=model, 
                                               input_tensor=img, 
                                               target_class=idx, 
                                               target_layer=layer)  
            #print(f"Size of heatmap is {heatmap.shape}")
            if args.save_saliency:
                # Process and save GradCam overlay to original img.
                upscaled_heatmap = process_heatmap(heatmap=heatmap, target_size=(320,320))
                overlayed_imgs = overlay_heatmap_on_img(original_img=img, heatmap=upscaled_heatmap, alpha=0.4)
                fig = visualize_heatmap(heatmap=overlayed_imgs, title=(f"Grad-CM for {id} on {target_name}"))
                save_path = os.path.join(output_dir, f"{args.model}/{target_name}_{id}.png")
                #print("Save path:", save_path)
                save_heatmap(fig, save_path)
            



            heatmap_vector = process_heatmap(heatmap=heatmap, 
                                      target_size=(10, 10), 
                                      normalize=True, 
                                      flatten=True,
                                      as_tensor=True)
            saliency_dict[target_name].append(heatmap_vector.cpu())

            centroids = compute_centroids(saliency_dict)
            distinctiveness = compute_distinctiveness(centroids)

            print("\nPer-class distinctiveness scores:")
            for cls, score in distinctiveness.items():
                print(f"  {cls}: {score:.4f}")

            fig = plot_distinctiveness(distinctiveness,
                                        save_path=os.path.join(output_dir, 'distinctiveness.png'))
            print(f"Saved distinctiveness plot to {output_dir}/distinctiveness.png")
               
    print("********** Finished saliency_maps script **********")   


if __name__ == "__main__":
    main()



