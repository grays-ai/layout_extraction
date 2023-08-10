from PIL import Image
import numpy as np


def rotation_pipeline(image, blocks):
	"""
	Checks if the image is rotated, and rotates it if it is. 
	"""
	rotated, angle = check_rotation(blocks)
	if rotated:
		image = rotate_image(image, angle)
	return image, rotated

	
def check_rotation(blocks):
	"""
	A block is rotated 90 degrees if the width is smaller than the height. Pick 10 random lines
	in blocks, and if a majority of them are rotated, then the page is rotated. 
	"""
	rotated = 0
	angle_90 = 0
	angle_270 = 0
	for i in range(100):
		line = blocks[np.random.randint(len(blocks))]
		if line['BlockType'] == 'LINE':
			print(line['Geometry']['BoundingBox'])
			if line['Geometry']['BoundingBox']['Width'] < line['Geometry']['BoundingBox']['Height']:
				rotated += 1
			if line['Geometry']['Polygon'][0]['Y'] > line['Geometry']['Polygon'][1]['Y']:
				angle_270 += 1
			else:
				angle_90 += 1
		if rotated > 10:
			
			angle = 270 if angle_270 > angle_90 else 90
			print(f"Rotated {angle} degrees clockwise")
			return True, angle
	return False, 0
			
def rotate_image(image, angle):
	return image.rotate(angle, expand=True)