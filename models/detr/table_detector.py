import requests as r
import mimetypes
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
import numpy as np
import boto3
import os
from pytesseract import Output
import pytesseract
import os
from dotenv import load_dotenv

load_dotenv()

ENDPOINT_URL_1= os.getenv("ENDPOINT_URL_1")
ENDPOINT_URL_2= os.getenv("ENDPOINT_URL_2")
ENDPOINT_URL_3= os.getenv("ENDPOINT_URL_3")
HF_TOKEN= os.getenv("HF_TOKEN")
from operator import itemgetter

# {'error': 'The size of tensor a (685) must match the size of tensor b (512) at non-singleton dimension 1'}


def detect_table(image: Image.Image):
	byte_stream = io.BytesIO()
	image.save(byte_stream, format='JPEG')
	byte_stream.seek(0) 
	image_data = byte_stream.read()

	headers= {
		"Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/jpeg"
	}
	response = r.post(ENDPOINT_URL_2, headers=headers, data=image_data)
	return response.json()

def extract_table(image: Image.Image):
	byte_stream = io.BytesIO()
	image.save(byte_stream, format='JPEG')
	byte_stream.seek(0) 
	image_data = byte_stream.read()
	headers= {
		"Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/jpeg"
	}
	response = r.post(ENDPOINT_URL_1, headers=headers, data=image_data)
	return response.json()

def ocr(image_array):
	image = Image.fromarray(np.uint8(image_array))
	
	# Create byte stream and save image as JPEG
	byte_stream = io.BytesIO()
	image.save(byte_stream, format='JPEG')
	byte_stream.seek(0)

	headers= {
		"Authorization": f"Bearer {HF_TOKEN}",
		"Content-Type": 'image/jpeg'
	}
	 # Get byte data
	image_byte_data = byte_stream.read()
	response = r.post(ENDPOINT_URL_1, headers=headers, data=image_byte_data)
	return response.json()


def only_ocr_image(image: Image.Image) -> str:
    """
    Performs Optical Character Recognition (OCR) on an image using pytesseract library.
    Args:
        image (PIL Image object): The image to perform OCR on.
    Returns:
        str: The recognized text from the image.
    """

    if not isinstance(image, Image.Image):
        raise TypeError(f'Expected a PIL Image, but got {type(image).__name__}')
    try:
        ocr_output = pytesseract.image_to_string(image, config='--psm 3', output_type=Output.STRING)
        return ocr_output
    except Exception as e:
        print(f'Error occurred during OCR process: {e}')
        return ''

def ocr_aws(image_array):

	session = boto3.Session(profile_name='jonahkaye')
	client = session.client('textract', region_name='us-east-1')	
	image = Image.fromarray(np.uint8(image_array))
	with io.BytesIO() as output:
		image.save(output, format="JPEG")
		image_bytes = output.getvalue()
	
	response = client.detect_document_text(Document={'Bytes': image_bytes})

	# Get the text blocks
	blocks = response['Blocks']

	# Initialize an empty string to store the OCR text
	ocr_text = ""
	previous_y = None
	word_count = 0

	# Append all detected text to the OCR string
	for block in blocks:
		if block['BlockType'] == 'WORD':
			# Get the X- and Y-coordinates of the bounding box
			current_y = block['Geometry']['BoundingBox']['Top']

			# If the Y-coordinate has changed significantly and X-coordinate has not, add a line break
			if previous_y is not None and abs(current_y - previous_y) > 0.01:
				ocr_text += '\n'

			ocr_text += block['Text'] + ' '
			previous_y = current_y
			word_count += 1

	if ocr_text.count('\n') >= word_count / 5:
		ocr_text = ocr_text.replace('\n', '')
	return ocr_text

def draw_table(prediction, image: Image.Image):
	# Load the image
	image = np.array(image)

	# Define color for the table box (in BGR)
	color = (0, 165, 255)  # Orange

	# Iterate over the prediction
	for item in prediction:
		# Get box coordinates
		xmin = item['box']['xmin']
		ymin = item['box']['ymin']
		xmax = item['box']['xmax']
		ymax = item['box']['ymax']
		label = item['label']

		# Only draw the box if the label is 'table'
		if label == 'table':
			# Draw rectangle on the image
			cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

			# Put label near the rectangle
			cv2.putText(image, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

	# Display the image with matplotlib (RGB order)
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.show()
	
	
def crop_table(prediction, image: Image.Image, padding= 100) -> Image.Image:
	# Load the image
	image = np.array(image)

	# Get box coordinates of the first detected table
	box = prediction[0]['box']
	xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']

	# Apply padding
	xmin = max(0, xmin - padding)
	ymin = max(0, ymin - padding)
	xmax = min(image.shape[1], xmax + padding)
	ymax = min(image.shape[0], ymax + padding)

	# Crop the image
	cropped_image = image[ymin:ymax, xmin:xmax]

	#convet to PIL image
	cropped_image = Image.fromarray(cropped_image)

	# Save the cropped image
	return cropped_image

def draw_rows(prediction, image: Image.Image):
	draw_boxes(prediction, image, only='table row')

def draw_columns(prediction, image: Image.Image):
	draw_boxes(prediction, image, only='table column')

def draw_boxes_only(prediction, image: Image.Image):
	draw_boxes(prediction, image, only='box')

def draw_boxes(prediction, image: Image.Image, only=None):
	# Load the image
	image = np.array(image)

	# Define colors for different box types (in BGR)
	colors = {
		'table row': (0, 255, 0),      # Green
		'table column': (255, 0, 0),   # Blue
		'box': (0, 165, 255)           # Orange
	}

	# Iterate over the prediction
	for item in prediction:
		# Get box coordinates
		xmin = item['box']['xmin']
		ymin = item['box']['ymin']
		xmax = item['box']['xmax']
		ymax = item['box']['ymax']
		label = item['label']

		# If only is specified and label does not match, continue
		if only is not None and label != only:
			continue

		# Choose color based on the label
		color = colors.get(label, (0, 0, 0))  # Use black as default color

		# Draw rectangle on the image
		cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

		# Put label near the rectangle
		cv2.putText(image, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

	# Display the image with matplotlib (RGB order)
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.show()
	
def intersect(row, column):
	return {'xmin': max(row['xmin'], column['xmin']),
			'ymin': max(row['ymin'], column['ymin']),
			'xmax': min(row['xmax'], column['xmax']),
			'ymax': min(row['ymax'], column['ymax'])}


def draw_cells(prediction, image: Image.Image):
	# Load the image
	image = np.array(image)

	# Separate rows and columns from prediction
	rows = [item for item in prediction if item['label'] == 'table row']
	columns = [item for item in prediction if item['label'] == 'table column']

	# Create a list to hold cell data
	cells = []

	# Iterate over the rows
	for row in rows:
		# Iterate over the columns
		for column in columns:
			# Compute the intersection of the row and column
			cell = intersect(row['box'], column['box'])
			
			# Add to cells
			cells.append(cell)

	# Define a list of colors for the boxes (in BGR)
	colors = [(0, 0, 255),  # Red
			  (0, 255, 0),  # Green
			  (255, 0, 0),  # Blue
			  (0, 255, 255),  # Cyan
			  (255, 0, 255),  # Magenta
			  (255, 255, 0)]  # Yellow

	# Iterate over the cells
	for i, cell in enumerate(cells):
		# Get box coordinates
		xmin = cell['xmin']
		ymin = cell['ymin']
		xmax = cell['xmax']
		ymax = cell['ymax']

		# Choose color based on the cell index
		color = colors[i % len(colors)]

		# Draw rectangle on the image
		cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

	# Display the image with matplotlib (RGB order)
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.show()
	return cells


def extract_table_from_predictions(predictions, image: Image.Image):

	# Load the image
	image = np.array(image)


	# Separate rows and columns from prediction
	rows = sorted([item for item in predictions if item['label'] == 'table row'], key=lambda x: x['box']['ymin'])
	columns = sorted([item for item in predictions if item['label'] == 'table column'], key=lambda x: x['box']['xmin'])

	# Create a 2D list (i.e., a list of lists) to hold the cell contents
	table = []

	# Iterate over the rows
	for row in rows:
		# Initialize a list to hold the current row's contents
		row_contents = []

		# Iterate over the columns
		for column in columns:
			# Compute the intersection of the row and column
			cell = intersect(row['box'], column['box'])

			# Crop cell from image
			xmin, ymin, xmax, ymax = cell['xmin'], cell['ymin'], cell['xmax'], cell['ymax']
			cropped_cell = image[ymin:ymax, xmin:xmax]

			# Extract cell content
			cropped_image = Image.fromarray(cropped_cell)

			cell_content = only_ocr_image(cropped_image)
			# cell_content = cell_content["predictions"]
			cell_content = cell_content.replace('\n', ' ')

			# Append cell content to the current row's contents
			row_contents.append(cell_content.strip())

		# Append the current row's contents to the table
		table.append(row_contents)

	# Convert the table (list of lists) into a pandas DataFrame
	df = pd.DataFrame(table)

	return df


def table_pipeline(image: Image.Image, idx):

	prediction = detect_table(image)
	print(f"For table {idx}: {prediction}")
	if len(prediction) != 0:
		draw_table(prediction, image)

		# Crop the table from the image
		cropped_image = crop_table(prediction, image)

		prediction = extract_table(cropped_image)
		print(prediction)
		#draw_columns(prediction, cropped_path)
		cells = draw_cells(prediction, cropped_image)
		df = extract_table_from_predictions(prediction, cropped_image)
		print(df.to_string())


if __name__ == '__main__':
	image_path = "/Users/jonahkaye/Desktop/startuping/grays-ai/layout_experimentation/images_testing/ayelet_table.jpeg"
	image = Image.open(image_path)
	table_pipeline(image, 1)




