# third_party/__init__.py

# Import specific functions to make them available directly from the package
from third_party.run_models import (
    parse_arguments, 
    prepare_data, 
    get_model, 
    load_checkpoint, 
    eval_model)

from third_party.utils import (
    get_target_layer,
    generate_gradcam_heatmap,
    process_heatmap,
    overlay_heatmap_on_img,
    visualize_heatmap,
    save_heatmap,
    compute_centroids,
    compute_distinctiveness,
    plot_distinctiveness,
    remove_prefix, 
    extract_study_id, 
    compute_accuracy, 
    comput_youden_idx, 
    compute_f1_score,
    class_distinctiveness,
    sum_up_distinctiveness,
    normalize_distinctiveness,
    plot_distinctiveness_boxplots
    )

# Define package-level variables
__version__ = '0.1.0'
__author__ = 'Fabricio'

# You could also run initialization code here
#print("Initializing third_party package")