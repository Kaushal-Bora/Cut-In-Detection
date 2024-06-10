from glob import glob
from xml.etree import ElementTree as et
from functools import reduce
import pandas as pd
import os
import shutil

folders = ['train', 'test', 'val']

def extract_xml_data(filename, folder):
	try:
		tree = et.parse(filename)
	except FileNotFoundError as error:
		return []

	path = filename.split("/")
	folder_name = path[-2]
	img = os.path.splitext(path[-1])[0] + '.jpg'
	img_name = folder_name + "_" + img

	root = tree.getroot()
	directory = folder + '/images/'
	height = root.find('size').find('height').text
	width = root.find('size').find('width').text
	objects = root.findall('object')
	data = []
	
	for obj in objects:
		label = obj.find('name').text
		if label not in ['car', 'bus', 'motorcycle', 'autorickshaw', 'truck']:
			continue
		bndbox = obj.find('bndbox')
		xmin = bndbox.find('xmin').text
		xmax = bndbox.find('xmax').text
		ymin = bndbox.find('ymin').text
		ymax = bndbox.find('ymax').text

		if label in ['car', 'autorickshaw']:
			data.append([img_name, width, height, 'medium', xmin, xmax, ymin, ymax])
		elif label == 'motorcycle':
			data.append([img_name, width, height, 'small', xmin, xmax, ymin, ymax])
		else:
			data.append([img_name, width, height, 'large', xmin, xmax, ymin, ymax])
		obj.clear()

	return data


def label_encoder(data):
	label = {'small':0, 'medium':1, 'large':2}
	return label[data]


def save_to_text(img_name, foldername, groupobj):
	path = foldername + os.path.splitext(img_name)[0] + '.txt'

	groupobj.get_group(img_name).set_index('img_name').to_csv(path, sep=' ', index=False, header=False)



for folder in folders:
	# Images Directory
	with open(folder + '.txt', 'r') as f:
		src = f.read().splitlines()

	dst = folder + '/images/'

	try:
	    os.makedirs(dst)
	    print(f"Creating new {folder} directory with {len(src)} images")
	except FileExistsError as error:
	    print(f"Directory named {folder} already exists")   

	for filepath in src:
		filepath = filepath + ".jpg"
		filename = "_".join(filepath.split("/")[1:])
		shutil.copyfile('JPEGImages/' + filepath, dst+filename)


	# Labels Directory
	with open(folder + '.txt', 'r') as f:
		src = f.read().splitlines()

	src_root = lambda x: "Annotations/" + os.path.splitext(x)[0] + '.xml'
	filenames = list(map(src_root, src))


	# xml_data_all = list(map(extract_xml_data, filenames, folder))
	xml_data_all = []
	for i in filenames:
		xml_data_all.append(extract_xml_data(i, folder))
	xml_data = reduce(lambda x, y: x+y, xml_data_all)

	df = pd.DataFrame(xml_data, columns=['img_name', 'width', 'height', 'label', 'xmin', 'xmax', 'ymin', 'ymax'])
	type_convert = ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']
	pd.options.display.max_rows = 500

	df[type_convert] = df[type_convert].astype(int)


	df['center_x'] = ((df['xmax'] + df['xmin'])/2)/df['width']
	df['center_y'] = ((df['ymax'] + df['ymin'])/2)/df['height']
	df['w'] = (df['xmax'] - df['xmin'])/df['width']
	df['h'] = (df['ymax'] - df['ymin'])/df['height']


	df['label_encoded'] = df['label'].apply(label_encoder)

	columns = ['img_name', 'label_encoded', 'center_x', 'center_y', 'w', 'h']
	group = df[columns].groupby('img_name')


	dst = folder + '/labels/'
	group_series = pd.Series(group.groups.keys())

	try:
	    os.makedirs(dst)
	    print(f"Creating new {folder} directory with {len(group.groups.keys())} labels")
	except FileExistsError as error:
	    print("Directory Exists")  

	group_series.apply(save_to_text, args=(dst, group))