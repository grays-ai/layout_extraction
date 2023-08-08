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


def ShowBoundingBox(draw,box,width,height,boxColor):
			 
	left = width * box['Left']
	top = height * box['Top'] 
	draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline=boxColor)   

def ShowSelectedElement(draw,box,width,height,boxColor):
			 
	left = width * box['Left']
	top = height * box['Top'] 
	draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],fill=boxColor)  

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

def process_text_analysis(client, image):

	# Convert PDF to image

	# Convert the image to bytes
	with io.BytesIO() as output:
		image.save(output, format="JPEG")
		image_bytes = output.getvalue()
		
	#detect_document_text
	response = client.detect_document_text(Document={'Bytes': image_bytes}) #, FeatureTypes=["TABLES"])
	return response['Blocks'], image
	
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
			
	result = has_column_structure(line_boxes, idx)
	print(f"for page {idx}, this is the result:  {result}")


	# Display the image
	return image

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
		bbox = line['Geometry']['BoundingBox']
		if avg_y in y_table:
			y_table[avg_y].append(line['Text'])
		else:
			y_table[avg_y] = [line['Text']]

	same_line_count = sum(len(entries) for key, entries in y_table.items() if len(entries) > 1)
	
	print(f"For page {idx}, same_line_count: {same_line_count}, len(lines): {len(lines)}")
	print(f"y_table: {y_table}")
	return same_line_count / len(lines) > 0.75

def main():

	session = boto3.Session(profile_name='jonahkaye')
	client = session.client('textract', region_name='us-east-1')
	document = "/Users/jonahkaye/Desktop/startuping/grays-ai/Fertility_Clinic/Rinehart_Records_2.pdf"
	images = convert_from_path(document)
	modified_images = []  # List to store modified images

	for idx, image in enumerate(images):
		blocks, image = process_text_analysis(client, image)
		modified_image = show_blocks(blocks, image, idx)
		modified_images.append(modified_image)

	#modified_images[0].save("/Users/jonahkaye/Desktop/startuping/grays-ai/eyenamics/gw.pdf", "PDF", resolution=100.0, save_all=True, append_images=modified_images[1:])


	# # Start searching a key value
	# while input('\n Do you want to search a value for a key? (enter "n" for exit) ') != 'n':
	# 	search_key = input('\n Enter a search key:')
	# 	print('The value is:', search_value(kvs, search_key))

	
if __name__ == "__main__":
	main()
