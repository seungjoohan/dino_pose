"""
Configuration for DINOv2 pose model training
"""

def get_default_configs():
    """
    Get default configurations for model, training, and preprocessing
    """
    config_dataset = {
        "train_images_dir": "/Users/seungjuhan/Desktop/vifive/vifive_pose_estimation/test_data/pose_datasets/threeD/train/images",
        "train_annotation_json": "/Users/seungjuhan/Desktop/vifive/vifive_pose_estimation/test_data/pose_datasets/threeD/train/annotation_3d_only_z.json",
        "val_images_dir": "/Users/seungjuhan/Desktop/vifive/vifive_pose_estimation/test_data/pose_datasets/threeD/valid/images",
        "val_annotation_json": "/Users/seungjuhan/Desktop/vifive/vifive_pose_estimation/test_data/pose_datasets/threeD/valid/annotation_3d_valid_only_z.json"
    }
    
    config_preproc = {
        # "is_background_swap": False,
        "pre_crop": True,
        "is_scale": True,
        "random_resize_min": 0.8,
        "random_resize_max": 1.2,
        "is_rotate": True,
        "rotate_min_degree": -40,
        "rotate_max_degree": 40,
        "is_flipping": True,
        "is_resize_shortest_edge": True,
        "is_crop": True,
        "is_occultation": False,
        "heatmap_std": 2.0
    }

    config_training = {
        "batch_size": 12,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "num_epochs": 30,
        "multiprocessing_num": 4,
        "print_freq": 10,
        "save_freq": 5,
        "checkpoint_dir": "dinov2_pose"
    }

    config_model = {
        "model_name": "facebook/dinov2-base",
        "num_keypoints": 24,
        "unfreeze_last_n_layers": 0,
        "output_heatmap_size": 48
    }
    
    return config_dataset, config_training, config_preproc, config_model