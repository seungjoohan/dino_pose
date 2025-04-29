from enum import Enum
import numpy as np
from pycocotools.coco import COCO

com_weights = np.array([
        0.081,
        0,
        0.140042,
        0.019204,
        0.015004,
        0.140042,
        0.019204,
        0.015004,
        0.18095,
        0.067334,
        0.036966,
        0.18095,
        0.067334,
        0.036966,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

class KeyPoints(Enum):
    """Enum representing keypoint indices"""
    TOP = 0
    NECK = 1
    RIGHT_SHOULDER = 2
    RIGHT_ELBOW = 3
    RIGHT_WRIST = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    RIGHT_HIP = 8
    RIGHT_KNEE = 9
    RIGHT_ANKLE = 10
    LEFT_HIP = 11
    LEFT_KNEE = 12
    LEFT_ANKLE = 13
    NOSE = 14
    RIGHT_EYE = 15
    RIGHT_EAR = 16
    LEFT_EYE = 17
    LEFT_EAR = 18
    SPINE = 19
    RIGHT_FINGER = 20
    RIGHT_TOE = 21
    LEFT_FINGER = 22
    LEFT_TOE = 23
    STERNUM = 24  # Additional computed point
    SACRUM = 25   # Additional computed point

class KeyPointConnections:
    """Lines and angles between keypoints"""
    links = [
        {'from': KeyPoints.TOP, 'to': KeyPoints.NECK, 'color': 'yellow'},
        {'from': KeyPoints.NECK, 'to': KeyPoints.RIGHT_SHOULDER, 'color': 'yellow'},
        {'from': KeyPoints.RIGHT_SHOULDER, 'to': KeyPoints.RIGHT_ELBOW, 'color': 'yellow'},
        {'from': KeyPoints.RIGHT_ELBOW, 'to': KeyPoints.RIGHT_WRIST, 'color': 'yellow'},
        {'from': KeyPoints.NECK, 'to': KeyPoints.LEFT_SHOULDER, 'color': 'yellow'},
        {'from': KeyPoints.LEFT_SHOULDER, 'to': KeyPoints.LEFT_ELBOW, 'color': 'yellow'},
        {'from': KeyPoints.LEFT_ELBOW, 'to': KeyPoints.LEFT_WRIST, 'color': 'yellow'},
        {'from': KeyPoints.NECK, 'to': KeyPoints.SPINE, 'color': 'pink'},
        {'from': KeyPoints.SPINE, 'to': KeyPoints.RIGHT_HIP, 'color': 'pink'},
        {'from': KeyPoints.RIGHT_HIP, 'to': KeyPoints.RIGHT_KNEE, 'color': 'pink'},
        {'from': KeyPoints.RIGHT_KNEE, 'to': KeyPoints.RIGHT_ANKLE, 'color': 'pink'},
        {'from': KeyPoints.SPINE, 'to': KeyPoints.LEFT_HIP, 'color': 'pink'},
        {'from': KeyPoints.LEFT_HIP, 'to': KeyPoints.LEFT_KNEE, 'color': 'pink'},
        {'from': KeyPoints.LEFT_KNEE, 'to': KeyPoints.LEFT_ANKLE, 'color': 'pink'},
        {'from': KeyPoints.TOP, 'to': KeyPoints.NOSE, 'color': 'green'},
        {'from': KeyPoints.NOSE, 'to': KeyPoints.RIGHT_EYE, 'color': 'green'},
        {'from': KeyPoints.RIGHT_EYE, 'to': KeyPoints.RIGHT_EAR, 'color': 'green'},
        {'from': KeyPoints.NOSE, 'to': KeyPoints.LEFT_EYE, 'color': 'green'},
        {'from': KeyPoints.LEFT_EYE, 'to': KeyPoints.LEFT_EAR, 'color': 'green'},
        {'from': KeyPoints.RIGHT_WRIST, 'to': KeyPoints.RIGHT_FINGER, 'color': 'blue'},
        {'from': KeyPoints.RIGHT_ANKLE, 'to': KeyPoints.RIGHT_TOE, 'color': 'blue'},
        {'from': KeyPoints.LEFT_WRIST, 'to': KeyPoints.LEFT_FINGER, 'color': 'blue'},
        {'from': KeyPoints.LEFT_ANKLE, 'to': KeyPoints.LEFT_TOE, 'color': 'blue'}
    ]
    
    angles = [
        {'first_start': KeyPoints.LEFT_SHOULDER, 'first_end': KeyPoints.LEFT_HIP,
         'second_start': '0', 'second_end': 'y'},
        {'first_start': KeyPoints.RIGHT_SHOULDER, 'first_end': KeyPoints.RIGHT_HIP,
         'second_start': '0', 'second_end': 'y'},
        {'first_start': KeyPoints.LEFT_SHOULDER, 'first_end': KeyPoints.RIGHT_SHOULDER,
         'second_start': '0', 'second_end': 'x'},
        {'first_start': KeyPoints.LEFT_HIP, 'first_end': KeyPoints.RIGHT_HIP,
         'second_start': '0', 'second_end': 'x'},
        {'first_start': KeyPoints.LEFT_KNEE, 'first_end': KeyPoints.LEFT_HIP,
         'second_start': KeyPoints.LEFT_HIP, 'second_end': KeyPoints.LEFT_SHOULDER},
        {'first_start': KeyPoints.RIGHT_KNEE, 'first_end': KeyPoints.RIGHT_HIP,
         'second_start': KeyPoints.RIGHT_HIP, 'second_end': KeyPoints.RIGHT_SHOULDER},
        {'first_start': KeyPoints.LEFT_ANKLE, 'first_end': KeyPoints.LEFT_KNEE,
         'second_start': KeyPoints.LEFT_KNEE, 'second_end': KeyPoints.LEFT_HIP},
        {'first_start': KeyPoints.RIGHT_ANKLE, 'first_end': KeyPoints.RIGHT_KNEE,
         'second_start': KeyPoints.RIGHT_KNEE, 'second_end': KeyPoints.RIGHT_HIP},
        {'first_start': KeyPoints.LEFT_ELBOW, 'first_end': KeyPoints.LEFT_SHOULDER,
         'second_start': KeyPoints.LEFT_SHOULDER, 'second_end': KeyPoints.LEFT_HIP},
        {'first_start': KeyPoints.RIGHT_ELBOW, 'first_end': KeyPoints.RIGHT_SHOULDER,
         'second_start': KeyPoints.RIGHT_SHOULDER, 'second_end': KeyPoints.RIGHT_HIP},
        {'first_start': KeyPoints.LEFT_WRIST, 'first_end': KeyPoints.LEFT_ELBOW,
         'second_start': KeyPoints.LEFT_ELBOW, 'second_end': KeyPoints.LEFT_SHOULDER},
        {'first_start': KeyPoints.RIGHT_WRIST, 'first_end': KeyPoints.RIGHT_ELBOW,
         'second_start': KeyPoints.RIGHT_ELBOW, 'second_end': KeyPoints.RIGHT_SHOULDER},
    ]

    @classmethod
    def get_skeleton_definition(cls):
        """Returns skeleton definition for visualization"""
        skeleton = []
        for link in cls.links:
            skeleton.append([link['from'].value, link['to'].value])
        return skeleton

def visualize_pose(image, keypoints, depths=None, threshold=0.5, figsize=(12, 12)):
    """
    Visualize pose keypoints on the image
    
    Args:
        image: PIL image or path to image
        keypoints: Array of keypoint coordinates (num_keypoints, 3) [x, y, confidence]
        depths: Array of keypoint depths (num_keypoints,) - optional
        threshold: Confidence threshold for visualization
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    # Convert to numpy array if needed
    img_np = np.array(image)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_np)
    
    # Skip if no keypoints
    if keypoints is None:
        return fig
    
    # Get visible keypoints
    mask = keypoints[:, 2] > threshold
    
    # Plot keypoints
    for i, point in enumerate(keypoints):
        if point[2] > threshold:
            # Circle size based on confidence
            circle_size = int(max(5, point[2] * 15))
            
            # Color based on depth if available
            if depths is not None:
                depth_normalized = min(1.0, depths[i] / depths[mask].max()) if depths[mask].size > 0 else 0.5
                color = (1 - depth_normalized, 0, depth_normalized)
            else:
                color = 'red'
            
            # Draw the keypoint
            circle = plt.Circle((point[0], point[1]), circle_size, color=color, alpha=0.7)
            ax.add_patch(circle)
            
            # Add keypoint index
            ax.text(point[0]+10, point[1]+10, f"{i}", fontsize=8, color='white', 
                   bbox=dict(facecolor='black', alpha=0.5))
    
    # Draw skeleton
    skeleton = KeyPointConnections.get_skeleton_definition()
    for link in KeyPointConnections.links:
        from_idx = link['from'].value
        to_idx = link['to'].value
        
        # Check if both keypoints are visible
        if keypoints[from_idx, 2] > threshold and keypoints[to_idx, 2] > threshold:
            x1, y1 = keypoints[from_idx, 0], keypoints[from_idx, 1]
            x2, y2 = keypoints[to_idx, 0], keypoints[to_idx, 1]
            ax.plot([x1, x2], [y1, y2], color=link['color'], linewidth=2, alpha=0.7)
    
    plt.axis('off')
    return fig

def read_annotation(annotation_path):
    coco = COCO(annotation_path)
    img_ids = coco.getImgIds()
    img_info = coco.loadImgs(img_ids)
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_ids))
    return img_info, anns