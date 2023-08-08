import argparse
import cv2
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import time
import warnings
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)

from pathlib import Path
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError


from .ditod.config import add_vit_config
from ...detr.table_detector import extract_table, crop_table, draw_cells, detect_table, draw_table, table_pipeline


def ShowBoundingBox(draw,box,width,height,boxColor):
			 
	left = width * box['Left']
	top = height * box['Top'] 
	draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline=boxColor)   

def process_file(input_file_path, output_file_path):
		input_file = Path(input_file_path)

		# Convert PDF to images
		#TODO this is where the temporary directory issues if they exist are happening
		with tempfile.TemporaryDirectory() as temp_dir:
			start_time = time.time()
			logging.info(f"Converting {input_file.name} pages to images...")
			
			images = convert_file_to_images(input_file, temp_dir)
			if images is None:
				return
			elapsed_time = time.time() - start_time
			logging.info(f"Image conversion Done! (Time elapsed: {elapsed_time:.2f} seconds)")
			logging.info(f"Total pages: {len(images)}")
			
			output_images = []
			for idx, img_pil in enumerate(images):
				#table_pipeline(img_pil, idx)
				start_ocr_time = time.time()  # record the start time of OCR
				output_image = layout_analysis(img_pil, idx)
				#cv2.imwrite(f"/Users/jonahkaye/Desktop/startuping/grays-ai/eyenamics/xxxx_{idx}.jpeg" , output_image)

				# output_image_pil = Image.fromarray(output_image, mode='RGB')
				elapsed_time_2 = time.time() - start_ocr_time
				logging.info(f"Layout Done! (Time elapsed: {elapsed_time_2:.2f} seconds)")
				output_images.append(output_image)

			output_images[0].save(output_file_path, save_all=True, append_images=output_images[1:])
				
def convert_file_to_images(input_file, temp_dir):
		if input_file.suffix == '.pdf':
			try:
				images = convert_from_path(str(input_file), output_folder=temp_dir, fmt='png', dpi=150, thread_count=8)
				return images
			except PDFPageCountError as e:
				logging.error(f"Error: {e}")
				return None

def layout_analysis(input_image: Image.Image, idx) -> Image.Image:
	parser = argparse.ArgumentParser(description="Detectron2 inference script")

	parser.add_argument(
		"--config-file",
		default="./models/dit/object_detection/publaynet_configs/cascade/cascade_dit_base.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument(
		"--opts",
		help="Modify config options using the command-line 'KEY VALUE' pairs",
		default=["MODEL.WEIGHTS", "./models/dit/publaynet_dit-b_cascade.pth"],
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
	img = np.array(input_image)

	md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
	if cfg.DATASETS.TEST[0]=='icdar2019_test':
		md.set(thing_classes=["table"])
	else:
		md.set(thing_classes=["text","title","list","table","figure"])

	output = predictor(img)["instances"]

	## Mask boxes whose corresponding pred_classes is 0
	text_class_mask = (output.pred_classes == 0)
	output.pred_boxes.tensor[text_class_mask] = 0.0

	# # Identify the class index for "title"
	# title_class_idx = md.get("thing_classes").index("title")

	# # Filter for boxes that belong to the "title" class
	# title_boxes = pred_boxes[pred_classes == title_class_idx] if pred_classes is not None else []

	# # Convert the image array to PIL Image for drawing
	# img_pil = Image.fromarray(img[:, :, ::-1])
	# draw = ImageDraw.Draw(img_pil)
	# width, height = img_pil.size

	# # Draw the boxes
	# for box in title_boxes:
	# 	prediction = {"box": {}}
	# 	prediction['box']['xmin'] = int(box[0])
	# 	prediction['box']['ymin'] = int(box[1])
	# 	prediction['box']['xmax'] = int(box[2])
	# 	prediction['box']['ymax'] = int(box[3])

	# 	# The provided ShowBoundingBox function seems to expect the boxes in a relative format.
	# 	# We can adjust this to handle absolute coordinates instead.
	# 	box_coords = {
	# 		'Left': prediction['box']['xmin'],
	# 		'Top': prediction['box']['ymin'],
	# 		'Width': prediction['box']['xmax'] - prediction['box']['xmin'],
	# 		'Height': prediction['box']['ymax'] - prediction['box']['ymin']
	# 	}
	# 	ShowBoundingBox(draw, box_coords, width, height, boxColor="red")

	# return img_pil


	# Extract bounding boxes, class names, and confidence scores
# 	pred_boxes = output.pred_boxes.tensor.cpu().numpy() if output.has("pred_boxes") else None
# 	pred_classes = output.pred_classes.cpu().numpy() if output.has("pred_classes") else None
# 	confidences = output.scores.cpu().numpy() if output.has("scores") else None

# 	all_classes = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes

# 	text_idx = all_classes.index('text')
# 	table_idx = all_classes.index('table')

# 	# Filter boxes
# 	text_boxes = []
# 	for i, c in enumerate(pred_classes):
# 		if c == text_idx:
# 			text_boxes.append(pred_boxes[i])
# 		# elif c == table_idx:

# 		# 	print("TABLE DETECTED")
# 		# 	table_path = f"/Users/jonahkaye/Desktop/startuping/grays-ai/eyenamics/table_{idx}.jpg"
# 		# 	cv2.imwrite(table_path, img)

# 		# 	prediction_table = detect_table(path_to_image=table_path)
# 		# 	print("Better table detector", prediction_table)
# 		# 	if prediction_table is not None:
# 		# 		draw_table(prediction_table, table_path)
# 		# 		cropped_table_path = crop_table(prediction_table, table_path)
# 		# 		# Detect cells on cropped table
# 		# 		prediction_extraction = extract_table(path_to_image=cropped_table_path)
# 		# 		cells = draw_cells(prediction_extraction, cropped_table_path)

# 		# 	if prediction_table is not None:
# 		# 		# Extract and crop the table 
# 		# 		prediction = {}
# 		# 		prediction[0] = {}
# 		# 		prediction[0]['box'] = {}
# 		# 		prediction[0]['box']['xmin'] = int(pred_boxes[i][0])  
# 		# 		prediction[0]['box']['ymin'] = int(pred_boxes[i][1]) 
# 		# 		prediction[0]['box']['xmax'] = int(pred_boxes[i][2])
# 		# 		prediction[0]['box']['ymax'] = int(pred_boxes[i][3])
# 		# 		cropped_table_path_2 = crop_table(prediction, table_path)
# 		# 		# Detect cells on cropped table
# 		# 		prediction_extract_2 = extract_table(path_to_image=cropped_table_path_2)
# 		# 		cells = draw_cells(prediction_extract_2, cropped_table_path)

# 	x_starts = [box[0] for box in text_boxes] 
# 	plt.hist(x_starts, bins=20)
# 	plt.title("Histogram of X Midpoints")
# 	plt.xlabel("X Pixel")
# 	plt.ylabel("Frequency")

# 	output_image_path = f"/Users/jonahkaye/Desktop/startuping/grays-ai/eyenamics/hist/test_{idx}.jpeg"
# 	plt.savefig(output_image_path)
# 	plt.clf() # Clear figure

# 	# Order everything by confidence scores in descending order
# 	if confidences is not None:
# 		sorted_indices = np.argsort(confidences)[::-1]  # get indices of sorted confidences in descending order
# 		pred_boxes = pred_boxes[sorted_indices]
# 		pred_classes = pred_classes[sorted_indices]
# 		confidences = confidences[sorted_indices]

#    # Define a sorting function for top-to-bottom
# 	def vertical_sort(box):
# 		return (box[1] + box[3]) / 2

# 	if pred_boxes is not None:
# 		x_midpoints = [(box[0] + box[2]) / 2 for box in pred_boxes]
# 		median_x = np.median(x_midpoints)
		
# 		right_column_indices = [i for i, x in enumerate(x_midpoints) if x > median_x]
# 		left_column_indices = [i for i, x in enumerate(x_midpoints) if x <= median_x]
		
# 		# Sort each column top to bottom
# 		right_column_indices.sort(key=lambda k: vertical_sort(pred_boxes[k]))
# 		left_column_indices.sort(key=lambda k: vertical_sort(pred_boxes[k]))

# 		# Combine the columns (right first, then left)
# 		sorted_indices = right_column_indices + left_column_indices
		
# 		pred_boxes = pred_boxes[sorted_indices]
# 		pred_classes = pred_classes[sorted_indices]
# 		confidences = confidences[sorted_indices]

	v = Visualizer(img[:, :, ::-1],
					md,
					scale=1.0,
					instance_mode=ColorMode.SEGMENTATION)
	result = v.draw_instance_predictions(output.to("cpu"))
	result_image = result.get_image()[:, :, ::-1]
	img_pil = Image.fromarray(result_image[:, :, ::-1])

	# step 6: save
	return img_pil

if __name__ == '__main__':
	process_file("/Users/jonahkaye/Desktop/startuping/grays-ai/eyenamics/xxxxQ.pdf", "/Users/jonahkaye/Desktop/startuping/grays-ai/eyenamics/xxxxV.pdf")

	# image = Image.open("/Users/jonahkaye/Desktop/startuping/grays-ai/layout_experimentation/images_testing/ayelet_table.jpeg")
	# image_np = np.array(image)  # Convert PIL Image to NumPy array

	# if len(image_np.shape) == 2:  # Now you can use shape
	# 	image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
	# 	print("ayelet")

	# image = Image.fromarray(image_np)
	# return_image = layout_analysis(image, 1)
	# cv2.imwrite("/Users/jonahkaye/Desktop/startuping/grays-ai/eyenamics/xxxx.pdf", return_image)
