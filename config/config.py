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
<<<<<<< Updated upstream
        "is_occultation": False,
        "heatmap_std": 0.5
=======
        "is_occultation": True,
        "heatmap_std": 1.5
>>>>>>> Stashed changes
    }

    config_training = {
        "batch_size": 32,
        "learning_rate": 3e-5,
        "weight_decay": 1e-6,
<<<<<<< Updated upstream
        "num_epochs": 5,
        "multiprocessing_num": 4,
        "save_freq": 5,
        "checkpoint_dir": "dinov2_test"
=======
        "num_epochs": 100,
        "multiprocessing_num": 4,
        "print_freq": 20,
        "save_freq": 10,
        "checkpoint_dir": "fastvit_lora"
>>>>>>> Stashed changes
    }

    config_model = {
        "model_name": "fastvit",
        "load_model": "fastvit_lora/best_model_100.pth",
        "num_keypoints": 24,
        "unfreeze_last_n_layers": 4,
        "use_lora": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "output_heatmap_size": 48
    }
    
    return config_dataset, config_training, config_preproc, config_model