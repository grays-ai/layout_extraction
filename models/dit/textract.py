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

def ShowBoundingBox(draw,box,width,height,boxColor):
			 
	left = width * box['Left']
	top = height * box['Top'] 
	draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline=boxColor)   

def ShowSelectedElement(draw,box,width,height,boxColor):
			 
	left = width * box['Left']
	top = height * box['Top'] 
	draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],fill=boxColor)  

def process_text_analysis(client, image):

	# Convert PDF to image

	# Convert the image to bytes
	with io.BytesIO() as output:
		image.save(output, format="JPEG")
		image_bytes = output.getvalue()
		
	#detect_document_text
	response = client.detect_document_text(Document={'Bytes': image_bytes}) #, FeatureTypes=["TABLES"])
	return response['Blocks'], image

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
		
		detected_text = ' '.join([box['Text'] for box in current_boxes])

		# 4. Create the bounding box (column box)
		column_box = {
			"Id": peak, # we might want to give this some ID
			"Detected": detected_text,
			"Type": "COLUMN",
			"Confidence": None, # average or use some other metric?
			"Relationships": None, # we can add more detailed relationship info if required
			"Geometry": {
				"Bounding Box": {
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

	# the set of boxes that are not in columns. O(n) lookup
	remaining_boxes = [box for box in line_boxes if box['Id'] not in boxes_inside_columns_set]
	remaining_text = ' '.join([box['Text'] for box in remaining_boxes])
	print(f"remaining_boxes: {len(remaining_boxes)}")
	print(f"overall boxes: {len(line_boxes)}")
	print(f"remaining_text: {remaining_text}")
	return columns, remaining_boxes


def find_column_peaks(hist, bin_edges, number_of_lines, distance=10):
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

	# Identify peaks in the histogram
	height = round(number_of_lines / 10)
	peaks, _ = find_peaks(hist, distance=distance, height=height)
	x_peaks = [bin_edges[peak] for peak in peaks]
	print(f"Peak x-values: {x_peaks}")

	return x_peaks

def generate_histogram(y_table, bin_width=0.01):
	"""
	Generates a histogram for the most common starting x-values.

	Parameters:
	- y_table (dict): Dictionary containing y averages as keys and their associated bounding boxes as values.
	- bin_width (float): The width of each bin, determining how "fuzzy" the histogram is.

	Returns:
	- hist (array): Values of the histogram.
	- bin_edges (array): The bin edges.
	"""

	# Extracting x_starts from y_table
	x_starts = []
	for _, boxes in y_table.items():
		for box in boxes:
			x_starts.append(box[0]['X'])

	# Create histogram
	hist, bin_edges = np.histogram(x_starts, bins=np.arange(0, 1 + bin_width, bin_width))

	# Plotting the histogram
	# plt.bar(bin_edges[:-1], hist, width=bin_width)
	# plt.xlabel('Starting x-value')
	# plt.ylabel('Count')
	# plt.title('Histogram of starting x-values')
	# plt.show()

	return hist, bin_edges

def calculate_avg_y(line):
	"""
	Calculate the average Y value of a line based on its polygon coordinates.
	"""
	y_values = [point['Y'] for point in line['Geometry']['Polygon']]
	# round to the nearest 100th
	return round(sum(y_values) / len(y_values), 2)

def has_column_structure(lines, idx):
	"""
	Determines if more than 75% of the lines on a given page are on the same line 
	as at least one other line by averaging Y coordinates and rounding to the nearest hundredth.
	"""
	y_table = {}

	for line in lines:
		avg_y = calculate_avg_y(line)
		bbox = line['Geometry']['Polygon']
		if avg_y in y_table:
			y_table[avg_y].append(bbox)
		else:
			y_table[avg_y] = [bbox]

	same_line_count = sum(len(entries) for key, entries in y_table.items() if len(entries) > 1)
	
	print(f"For page {idx}, same_line_count: {same_line_count}, len(lines): {len(lines)}")
	#print(f"y_table: {y_table}")
	is_column = same_line_count / len(lines) > 0.75
	return is_column, y_table, len(lines)

def show_blocks(blocks, image, idx):
	width, height =image.size    
   
	line_boxes = []  # Initialize outside the loop
	draw=ImageDraw.Draw(image)

	# Create image showing bounding box/polygon the detected lines/text
	for block in blocks:

		# Draw bounding boxes for different detected response objects
		if block['BlockType'] == "PAGE":
			print("NEW PAGE")            
		if block['BlockType'] == 'TABLE':
			ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'blue')
		if block['BlockType'] == 'CELL':
			ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'green')
		if block['BlockType'] == 'LINE':
			ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'orange')
			line_boxes.append(block)
		if block['BlockType'] == 'SELECTION_ELEMENT':
			if block['SelectionStatus'] =='SELECTED':
				ShowSelectedElement(draw, block['Geometry']['BoundingBox'],width,height, 'blue')
			
	is_column, y_table, number_of_lines = has_column_structure(line_boxes, idx)
	if is_column:
		hist, bin_edges = generate_histogram(y_table)
		peaks = find_column_peaks(hist, bin_edges, number_of_lines)
		columns = find_columns(peaks, line_boxes)
		for column in columns:
			ShowBoundingBox(draw, column['Geometry']['Bounding Box'], width, height, 'red')

	return image


def main():

	session = boto3.Session(profile_name='jonahkaye')
	client = session.client('textract', region_name='us-east-1')
	document = "/Users/jonahkaye/Desktop/startuping/grays-ai/eyenamics/xxxx.pdf"
	images = convert_from_path(document)
	modified_images = []  # List to store modified images

	for idx, image in enumerate(images):
		blocks, image = process_text_analysis(client, image)
		modified_image = show_blocks(blocks, image, idx)
		modified_images.append(modified_image)


	modified_images[0].save("/Users/jonahkaye/Desktop/startuping/grays-ai/eyenamics/xxxxK.pdf", "PDF", resolution=100.0, save_all=True, append_images=modified_images[1:])


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

def best_cluster_count(y_table):
	"""
	Determines the best cluster count for x-start values.

	Parameters:
	- y_table (dict): Dictionary containing y averages as keys and their associated bounding boxes as values.

	Returns:
	- best_k (int): The optimal number of clusters.
	- best_score (float): Silhouette score of the best clustering.
	- best_labels (list): Cluster labels for the best clustering.
	"""
	# Extracting x_starts from y_table
	x_starts = []
	boxes = []

	for _, boxes in y_table.items():
		for box in boxes:
			x_starts.append(box[0]['X'])
			boxes.append(box)


	# Find two most frequent counts, excluding lines with just one value
	line_counts = Counter([len(lines) for lines in y_table.values() if len(lines) > 1])
	most_common_counts = [item[0] for item in line_counts.most_common(2)]
	print(f"most_common_counts: {most_common_counts}")

	# Cluster and evaluate for both counts
	best_score = -1  # Silhouette scores range from -1 to 1
	best_k = None
	best_labels = None

	for k in most_common_counts:
		kmeans = KMeans(n_clusters=k)
		labels = kmeans.fit_predict(np.array(x_starts).reshape(-1, 1))
		score = silhouette_score(np.array(x_starts).reshape(-1, 1), labels)
		print(f"for k = {k}, score = {score}")
		
		if score > best_score:
			best_score = score
			best_k = k
			best_labels = labels

	return best_k, best_score, best_labels, boxes


def column_bounding_boxes(labels, boxes):
	"""
	Given a list of cluster labels and boxes, compute the bounding box for each cluster.

	Parameters:
	- labels (list): Cluster labels for each bounding box.
	- boxes (list): List of bounding boxes.

	Returns:
	- List of bounding boxes, one for each cluster.
	"""
	# Dictionary to hold bounding box data for each cluster
	clusters = {}
	for label, box in zip(labels, boxes):
		if label not in clusters:
			clusters[label] = {'min_left': box['Left'],
							   'max_right': box['Left'] + box['Width'],
							   'min_top': box['Top'],
							   'max_bottom': box['Top'] + box['Height']}
		else:
			clusters[label]['min_left'] = min(clusters[label]['min_left'], box['Left'])
			clusters[label]['max_right'] = max(clusters[label]['max_right'], box['Left'] + box['Width'])
			clusters[label]['min_top'] = min(clusters[label]['min_top'], box['Top'])
			clusters[label]['max_bottom'] = max(clusters[label]['max_bottom'], box['Top'] + box['Height'])

	# Convert data to list of bounding boxes
	column_boxes = []
	for _, vals in clusters.items():
		column_boxes.append({'Left': vals['min_left'],
							 'Top': vals['min_top'],
							 'Width': vals['max_right'] - vals['min_left'],
							 'Height': vals['max_bottom'] - vals['min_top']})

	return column_boxes
	
if __name__ == "__main__":
	main()
