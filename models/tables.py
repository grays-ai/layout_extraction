import webbrowser, os
import json
import boto3
import io
from io import BytesIO
import sys
from pprint import pprint

def ShowBoundingBox(draw,box,width,height,boxColor, lineThickness=10):
			 
	left = width * box['Left']
	top = height * box['Top'] 
	draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline=boxColor, width=lineThickness
)   

def get_rows_columns_map(table_result, blocks_map, word_children):
	rows = {}
	scores = []
	for relationship in table_result['Relationships']:
		if relationship['Type'] == 'CHILD':
			for child_id in relationship['Ids']:
				cell = blocks_map[child_id]
				if cell['BlockType'] == 'CELL':
					row_index = cell['RowIndex']
					col_index = cell['ColumnIndex']
					if 'Relationships' in cell:
						word_children.extend(cell['Relationships'][0]['Ids'])
					if row_index not in rows:
						# create new row
						rows[row_index] = {}
					
					# get confidence score
					#scores.append(str(cell['Confidence']))
						
					# get the text value
					rows[row_index][col_index] = get_text(cell, blocks_map)
	return rows, scores, word_children


def get_text(result, blocks_map):
	text = ''
	if 'Relationships' in result:
		for relationship in result['Relationships']:
			if relationship['Type'] == 'CHILD':
				for child_id in relationship['Ids']:
					word = blocks_map[child_id]
					if word['BlockType'] == 'WORD':
						if "," in word['Text'] and word['Text'].replace(",", "").isnumeric():
							text += '"' + word['Text'] + '"' + ' '
						else:
							text += word['Text'] + ' '
					if word['BlockType'] == 'SELECTION_ELEMENT':
						if word['SelectionStatus'] =='SELECTED':
							text +=  'X '
	return text


def generate_table_csv(table_result, blocks_map, table_index, word_children):
	rows, scores, word_children = get_rows_columns_map(table_result, blocks_map, word_children)

	table_id = 'Table_' + str(table_index)
	
	# get cells.
	csv = 'THIS IS A CSV TABLE: {0}\n\n'.format(table_id)

	for row_index, cols in rows.items():
		for col_index, text in cols.items():
			col_indices = len(cols.items())
			csv += '{}'.format(text) + ","
		csv += '\n'
		
	return csv, word_children


def get_table_csv_results(blocks, draw, width, height) -> (str, bool, list):

	blocks_map = {}
	table_blocks = []
	words_to_lines_map = {}
	lines_hash = {}	

	for block in blocks:
		blocks_map[block['Id']] = block
		if block['BlockType'] == "TABLE":
			table_blocks.append(block)
			ShowBoundingBox(draw, block['Geometry']['BoundingBox'], width, height, 'blue')
		elif block['BlockType'] == 'LINE':
			lines_hash[block['Id']] = block
			for relationship in block['Relationships']:
				if relationship['Type'] == 'CHILD':
					for child_id in relationship['Ids']:
						words_to_lines_map[child_id] = block['Id']

	if len(table_blocks) <= 0:
		return "", False, blocks

	csv = ''
	word_children = []
	for index, table in enumerate(table_blocks):
		csv_string, word_children = generate_table_csv(table, blocks_map, index +1, word_children)
		csv += csv_string
		csv += '\n\n'
	
	for child in word_children:
		if child in words_to_lines_map:
			line = words_to_lines_map[child]
			if line in lines_hash:
				#remove from lines_hash
				del lines_hash[line]

	blocks_list = list(lines_hash.values())

	return csv, True, blocks_list