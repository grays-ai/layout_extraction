import boto3
import io
import time
from PIL import Image, ImageDraw
import sys
import re
import json
from collections import defaultdict
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)

from .table_extraction import get_table_csv_results
from .image_rotation import rotation_pipeline

def ShowBoundingBox(draw,box,width,height,boxColor, lineThickness=10):
			 
	left = width * box['Left']
	top = height * box['Top']
	draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline=boxColor, width=lineThickness
)   

def ShowSelectedElement(draw,box,width,height,boxColor):
			 
	left = width * box['Left']
	top = height * box['Top'] 
	draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],fill=boxColor)  

def aws_get_lines(client, image):

	# Convert PDF to image

	# Convert the image to bytes
	with io.BytesIO() as output:
		image.save(output, format="JPEG")
		image_bytes = output.getvalue()
		
	#detect_document_text
	response = client.detect_document_text(Document={'Bytes': image_bytes})
	return response['Blocks']

def aws_extract_tables(client, image):

	# Convert PDF to image

	# Convert the image to bytes
	with io.BytesIO() as output:
		image.save(output, format="JPEG")
		image_bytes = output.getvalue()
		
	#detect_document_text
	response = client.analyze_document(Document={'Bytes': image_bytes}, FeatureTypes=["TABLES"])
	return response['Blocks']

#########################

def find_columns(peaks, line_boxes, threshold=0.02):

	"""
	Algorithm to find the columns given the starting x coordinates of the peaks. We start by finding the first box, 
	the leftmost box, the rightmost box, and the end box, for all boxes that are close to the peak.

	Then we constuct the column box by using the leftmost and rightmost boxes to find the width, and the first and
	end boxes to find the height. Then we find any boxes inside the column that werent next to the peak. 

	We do this for each peak, and then subtract all boxes that were in columns from the original list of boxes to find
	the boxes that remain that were outside the column. 

	Order of transcription should be: remaining boxes, COLUMNn1, Columnn2, .... 

	"""
	columns = []
	boxes_inside_columns_set = set()  # A set to keep track of box IDs already in columns
	columns_text = ""

	for idx, peak in enumerate(peaks):
		# 1. Initialize values
		starting_box = None
		ending_box = None
		rightmost_x = float('-inf')
		rightmost_box = None
		leftmost_x = float('inf')
		leftmost_box = None

		# Iterate over line_boxes to find required boxes in one pass
		for box in line_boxes:
			if abs(peak - box['Geometry']['Polygon'][0]['X']) <= threshold:
				if not starting_box:
					starting_box = box

				ending_box = box

				if box['Geometry']['Polygon'][0]['X'] < leftmost_x:
					leftmost_x = box['Geometry']['Polygon'][0]['X']
					leftmost_box = box

				if box['Geometry']['Polygon'][2]['X'] > rightmost_x:
					rightmost_x = box['Geometry']['Polygon'][2]['X']
					rightmost_box = box
			
		if not starting_box:  # No boxes were close to the peak
			continue

		# one more iteration to find all the boxes in the column
		current_boxes_and_ids = [
			(box, box['Id']) for box in line_boxes
			if box['Geometry']['Polygon'][0]['X'] >= leftmost_box['Geometry']['Polygon'][0]['X'] and
			box['Geometry']['Polygon'][2]['X'] <= rightmost_box['Geometry']['Polygon'][2]['X']
		]

		current_boxes, current_box_ids = zip(*current_boxes_and_ids)

		boxes_inside_columns_set.update(current_box_ids)
		
		detected_text = ' '.join([box.get('Text', '') for box in current_boxes])

		# 4. Create the bounding box (column box)
		column_box = {
			"Id": peak, # we might want to give this some ID
			"Text": detected_text,
			"BlockType": "COLUMN",
			"Confidence": 0.0, # average or use some other metric?
			"Geometry": {
				"BoundingBox": {
					'Width': rightmost_box['Geometry']['Polygon'][2]['X'] - leftmost_box['Geometry']['Polygon'][0]['X'],
					'Height': ending_box['Geometry']['Polygon'][3]['Y'] - starting_box['Geometry']['Polygon'][0]['Y'],
					'Left': leftmost_box['Geometry']['Polygon'][0]['X'],
					'Top': starting_box['Geometry']['Polygon'][0]['Y']
				},
			   	"Polygon": [
					{'X': peak, 'Y': starting_box['Geometry']['Polygon'][0]['Y']},  # Top-left
					{'X': rightmost_box['Geometry']['Polygon'][1]['X'], 'Y': rightmost_box['Geometry']['Polygon'][1]['Y']},  # Top-right
					{'X': rightmost_box['Geometry']['Polygon'][2]['X'], 'Y': ending_box['Geometry']['Polygon'][2]['Y']},    # Bottom-right
					{'X': peak, 'Y': ending_box['Geometry']['Polygon'][3]['Y']}      # Bottom-left
				]
			}
		}
		columns.append(column_box)
		columns_text += detected_text + "\n\n"

	# the set of boxes that are not in columns. O(n) lookup
	remaining_boxes = [box for box in line_boxes if box['Id'] not in boxes_inside_columns_set]
	print(f"remaining_boxes: {len(remaining_boxes)}")
	print(f"overall boxes: {len(line_boxes)}")

	remaining_text = ' '.join([box.get('Text', '') for box in remaining_boxes])
	return columns, remaining_boxes, remaining_text, columns_text


def find_column_peaks_and_draw_lines(line_boxes, number_of_lines, draw, width, height, distance=10, bin_width=0.01):
	"""
	Identify the peaks in the histogram using a simple distance threshold. 

	Parameters:
	- hist (array): Values of the histogram.
	- bin_edges (array): The bin edges.
	- distance (int): Minimum distance between peaks. Adjust as necessary.
	- height (int or None): Minimum height of peaks. Use to filter out noise.

	Returns:
	- column_ranges (list of tuples): Each tuple corresponds to the range (start, end) of a column's x-values.
	"""

	x_starts = []
	for box in line_boxes:
		x_starts.append(box['Geometry']['Polygon'][0]['X'])
		ShowBoundingBox(draw, box['Geometry']['BoundingBox'], width, height, 'orange')

	# Create histogram
	hist, bin_edges = np.histogram(x_starts, bins=np.arange(0, 1 + bin_width, bin_width))
	# Identify peaks in the histogram
	height = round(number_of_lines / 10)
	peaks, _ = find_peaks(hist, distance=distance, height=height)
	x_peaks = [bin_edges[peak] for peak in peaks]
	print(f"Peak x-values: {x_peaks}")

	return x_peaks


############

def calculate_avg_y(line):
	"""
	Calculate the average Y value of a line based on its polygon coordinates.
	"""
	y_values = [point['Y'] for point in line['Geometry']['Polygon']]
	# round to the nearest 100th
	return round(sum(y_values) / len(y_values), 2)

def has_column_table_structure(lines, idx, threshold=0.75):
	"""
	Determines if more than 75% of the lines on a given page are on the same line 
	as at least one other line by averaging Y coordinates and rounding to the nearest hundredth.

	Or if any single line has more than 6 lines on that line. 
	"""
	y_table = {}

	for line in lines:
		avg_y = calculate_avg_y(line)
		bbox = line['Geometry']['Polygon']
		if avg_y in y_table:
			y_table[avg_y].append(bbox)
		else:
			y_table[avg_y] = [bbox]

	same_line_count = 0
	has_greater_than_six = False

	for entries in y_table.values():
		if len(entries) > 1:
			same_line_count += len(entries)
		if len(entries) > 6:
			has_greater_than_six = True
			break
	
	print(f"For page {idx}, same_line_count: {same_line_count}, len(lines): {len(lines)}")
	#print(f"y_table: {y_table}")
	if len(lines) == 0:
		return False, 0
	
	is_column = (same_line_count / len(lines) > threshold) or has_greater_than_six
	return is_column, len(lines)

############

def column_extraction(line_boxes, draw, width, height, number_of_lines, idx):
	print(f"Page {idx}: Column extraction")
	peaks = find_column_peaks_and_draw_lines(line_boxes, number_of_lines, draw, width, height)
	columns, remaining_boxes, remaining_text, detected_text= find_columns(peaks, line_boxes)
	for column in columns:
		ShowBoundingBox(draw, column['Geometry']['BoundingBox'], width, height, 'red')

	return columns, remaining_boxes, remaining_text, detected_text

def table_extraction(draw, width, height, client, image, idx) -> (str, bool, list):
	print(f"Page {idx}: Table extraction")
	blocks = aws_extract_tables(client, image)

	return get_table_csv_results(blocks, draw, width, height)


def regular_extraction(line_boxes, draw, width, height, idx):
	for box in line_boxes:
		ShowBoundingBox(draw, box['Geometry']['BoundingBox'], width, height, 'yellow')
	print(f"Page {idx}: Regular extraction")



def show_and_extract_blocks(client, image, idx):

	blocks = aws_get_lines(client, image)
	#checks if image is rotated and if so, rotates it 90 degrees
	image, rotated = rotation_pipeline(image, blocks)
	if rotated:
		# call get lines again on the new image
		blocks = aws_get_lines(client, image)
	width, height =image.size   
	# Get the text block 
   
	line_boxes = []  # Initialize outside the loop
	table_boxes = []
	draw=ImageDraw.Draw(image)

	# Create image showing bounding box/polygon the detected lines/text
	for block in blocks:
		if block['BlockType'] == 'LINE':
			line_boxes.append(block)

	is_col_table, number_of_lines = has_column_table_structure(line_boxes, idx, threshold=0.65)
	if is_col_table:
		tables_string, was_tables_found, lines_boxes_no_table = table_extraction(draw, width, height, client, image, idx)
		if was_tables_found:
			is_col, number_of_lines_no_table = has_column_table_structure(lines_boxes_no_table, idx, threshold=0.65)
			if is_col:
				columns, remaining_boxes, remaining_text, column_text = column_extraction(lines_boxes_no_table, draw, width, height, number_of_lines_no_table, idx)
				print("REMAINING TEXT:", remaining_text)
				print("DETECTED TEXT:", column_text)
			else:
				regular_extraction(lines_boxes_no_table, draw, width, height, idx)
			print("TABLE TEXT:", tables_string)
		else:
			# extract according to the original structure that was found. 
			columns, remaining_boxes, remaining_text, column_text = column_extraction(lines_boxes_no_table, draw, width, height, number_of_lines, idx)
			print("REMAINING TEXT:", remaining_text)
			print("DETECTED TEXT:", column_text)
	else:
		regular_extraction(line_boxes, draw, width, height, idx)

	return image


def main():

	session = boto3.Session(profile_name='jonahkaye')
	client = session.client('textract', region_name='us-east-1')
	document = "/Users/jonahkaye/Desktop/startuping/grays-ai/Fertility_Clinic/sana.pdf"
	images = convert_from_path(document)
	modified_images = []  # List to store modified images
	

	for idx, image in enumerate(images):
		modified_image = show_and_extract_blocks(client, image, idx)
		modified_images.append(modified_image)


	modified_images[0].save("/Users/jonahkaye/Desktop/startuping/grays-ai/Fertility_Clinic/sana2k.pdf", "PDF", resolution=100.0, save_all=True, append_images=modified_images[1:])


# Displays information about a block returned by text detection and text analysis
def DisplayBlockInformation(block):
	print('Id: {}'.format(block['Id']))
	if 'Text' in block:
		print('    Detected: ' + block['Text'])
	print('    Type: ' + block['BlockType'])
   
	if 'Confidence' in block:
		print('    Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")

	if block['BlockType'] == 'CELL':
		print("    Cell information")
		print("        Column:" + str(block['ColumnIndex']))
		print("        Row:" + str(block['RowIndex']))
		print("        Column Span:" + str(block['ColumnSpan']))
		print("        RowSpan:" + str(block['ColumnSpan']))    
	
	if 'Relationships' in block:
		print('    Relationships: {}'.format(block['Relationships']))
	print('    Geometry: ')
	print('        Bounding Box: {}'.format(block['Geometry']['BoundingBox']))
	print('        Polygon: {}'.format(block['Geometry']['Polygon']))
	
	if block['BlockType'] == "KEY_VALUE_SET":
		print ('    Entity Type: ' + block['EntityTypes'][0])
	
	if block['BlockType'] == 'SELECTION_ELEMENT':
		print('    Selection element detected: ', end='')

		if block['SelectionStatus'] =='SELECTED':
			print('Selected')
		else:
			print('Not selected')    
	
	if 'Page' in block:
		print('Page: ' + block['Page'])
	print()


if __name__ == "__main__":
	main()
