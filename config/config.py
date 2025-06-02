"""
Configuration for DINOv2 pose model training
"""

def get_default_configs():
    """
    Get default configurations for model, training, and preprocessing
    """
    config_dataset = {
        "train_images_dir": "/Users/seunghan/Desktop/vifive/vifive_pose_estimation/test_data/pose_datasets/threeD/train/images",
        "train_annotation_json": "/Users/seunghan/Desktop/vifive/vifive_pose_estimation/test_data/pose_datasets/threeD/train/annotation_3d_only_z.json",
        "val_images_dir": "/Users/seunghan/Desktop/vifive/vifive_pose_estimation/test_data/pose_datasets/threeD/valid/images",
        "val_annotation_json": "/Users/seunghan/Desktop/vifive/vifive_pose_estimation/test_data/pose_datasets/threeD/valid/annotation_3d_valid_only_z.json"
    }
    
    config_preproc = {
        # "is_background_swap": False,
        "pre_crop": True,
        "is_scale": True,
        "random_resize_min": 0.7,
        "random_resize_max": 1.3,
        "is_rotate": True,
        "rotate_min_degree": -45,
        "rotate_max_degree": 45,
        "is_flipping": True,
        "is_resize_shortest_edge": True,
        "is_crop": True,
        "is_occultation": False,
        "heatmap_std": 1.5
    }

    config_training = {
        "batch_size": 32,
        "learning_rate": 1e-5,
        "weight_decay": 5e-7,
        "num_epochs": 50,
        "multiprocessing_num": 4,
        "print_freq": 10,
        "save_freq": 5,
        "checkpoint_dir": "dinov2_pose_small"
    }

    config_model = {
        "model_name": "facebook/dinov2-small",
        "load_model": "",
        "num_keypoints": 24,
        "unfreeze_last_n_layers": 8,
        "use_lora": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "output_heatmap_size": 48
    }
    
    return config_dataset, config_training, config_preproc, config_model