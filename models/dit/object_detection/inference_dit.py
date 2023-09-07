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
from detectron2.structures import Instances



from ditod import add_vit_config
#from ...detr.table_detector import extract_table, crop_table, draw_cells, detect_table, draw_table, table_pipeline


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
	img = np.array(input_image)

	md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
	if cfg.DATASETS.TEST[0]=='icdar2019_test':
		md.set(thing_classes=["table"])
	else:
		md.set(thing_classes=["text","title","list","table","figure"])

	output = predictor(img)["instances"]

	 ## Only keep boxes whose corresponding pred_classes is for "title"
	title_class_index = md.get("thing_classes").index("title")

	title_class_mask = (output.pred_classes == title_class_index)
	
	# Create a new Instances object with only "title" predictions
	title_instances = Instances(output.image_size)
	title_instances.pred_boxes = output.pred_boxes[title_class_mask]
	title_instances.pred_classes = output.pred_classes[title_class_mask]
	title_instances.scores = output.scores[title_class_mask]

	v = Visualizer(img[:, :, ::-1],
				   md,
				   scale=1.0,
				   instance_mode=ColorMode.SEGMENTATION)
	result = v.draw_instance_predictions(title_instances.to("cpu"))
	result_image = result.get_image()[:, :, ::-1]
	img_pil = Image.fromarray(result_image[:, :, ::-1])

	# step 6: save
	return img_pil

if __name__ == '__main__':
	#process_file("./Patient1468183PDF545177.pdf", "./1468183PDF545177.pdf")


	img = layout_analysis(Image.open("./69ab684d-Patient1569838PDF2294173.png"), 1)
	#show the image
	img.show()