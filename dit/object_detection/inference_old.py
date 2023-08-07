import argparse

import cv2

from ditod import add_vit_config

import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
import matplotlib.pyplot as plt
import numpy as np




def main():
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    parser.add_argument(
        "--image_path",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file_name",
        help="Name of the output visualization file.",
        type=str,
    )
    parser.add_argument(
        "--config-file",
        default="./dit/object_detection/publaynet_configs/cascade/cascade_dit_base.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "./dit/publaynet_dit-b_cascade.pth"],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    # Step 2: add model weights URL to config
    cfg.merge_from_list(args.opts)
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)
    
    # Step 5: run inference
    img = cv2.imread(args.image_path)

    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if cfg.DATASETS.TEST[0]=='icdar2019_test':
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=["text","title","list","table","figure"])

    output = predictor(img)["instances"]

    # Enable interactive mode in matplotlib
    plt.ion()

    # Define a larger figure size
    fig = plt.figure(figsize=(10, 10))

    # Extract bounding boxes, class names, and confidence scores
    pred_boxes = output.pred_boxes.tensor.cpu().numpy() if output.has("pred_boxes") else None
    pred_classes = output.pred_classes.cpu().numpy() if output.has("pred_classes") else None
    confidences = output.scores.cpu().numpy() if output.has("scores") else None

#     # Order everything by confidence scores in descending order
#     if confidences is not None:
#         sorted_indices = np.argsort(confidences)[::-1]  # get indices of sorted confidences in descending order
#         pred_boxes = pred_boxes[sorted_indices]
#         pred_classes = pred_classes[sorted_indices]
#         confidences = confidences[sorted_indices]

#    # Define a sorting function for top-to-bottom
#     def vertical_sort(box):
#         return (box[1] + box[3]) / 2

#     if pred_boxes is not None:
#         x_midpoints = [(box[0] + box[2]) / 2 for box in pred_boxes]
#         median_x = np.median(x_midpoints)
        
#         right_column_indices = [i for i, x in enumerate(x_midpoints) if x > median_x]
#         left_column_indices = [i for i, x in enumerate(x_midpoints) if x <= median_x]
        
#         # Sort each column top to bottom
#         right_column_indices.sort(key=lambda k: vertical_sort(pred_boxes[k]))
#         left_column_indices.sort(key=lambda k: vertical_sort(pred_boxes[k]))

#         # Combine the columns (right first, then left)
#         sorted_indices = right_column_indices + left_column_indices
        
#         pred_boxes = pred_boxes[sorted_indices]
#         pred_classes = pred_classes[sorted_indices]
#         confidences = confidences[sorted_indices]

#     # Display each box
#     for idx, box in enumerate(pred_boxes):
#         # Clone the image to draw on it
#         img_clone = img.copy()
        
#         # Draw the individual bounding box
#         cv2.rectangle(img_clone, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        
#         class_name = md.thing_classes[pred_classes[idx]] if pred_classes is not None else "unknown"
#         confidence = confidences[idx] if confidences is not None else "unknown"
        
#         # Display the image with matplotlib
#         plt.imshow(cv2.cvtColor(img_clone, cv2.COLOR_BGR2RGB))
#         plt.title(f"Bounding Box for {class_name} (Confidence: {confidence:.4f})")
#         plt.axis('off')
#         plt.draw()
#         plt.pause(1)  # pause for 2 seconds before showing the next image

#     # Disable interactive mode when done
#     plt.ioff()
#     plt.show()

    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(output.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]

    # step 6: save
    cv2.imwrite(args.output_file_name, result_image)

if __name__ == '__main__':
    main()