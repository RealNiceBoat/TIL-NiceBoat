from glob import iglob
from pathlib import Path
import numpy as np
from multiprocessing import Pool
import PIL,json,pickle
'''
TILSequence and TILPickle both implement Sequence, a object for easily loading batches used in model training provided by the Tensorflow.
One loads it directly from the img directory and annotation, the other loads it from a pro-processed pickle of the dataset.
'''


'''
#if using pyTorch, equivalent is:
from torch.utils.data import Dataset
class TILSequence(Dataset):
	...
'''
from tensorflow.python.keras.utils.data_utils import Sequence
class TILSequence(Sequence):
	def __init__(self, img_dir, meta_file, batch_size, augmenter, input_size, label_encoder, preprocessor, testmode=False):
		self.batch_size = batch_size
		self.augmenter = augmenter
		self.input_res = (*input_size[:2][::-1],input_size[2])
		self.label_encoder = label_encoder
		self.preprocessor = preprocessor
		self.testmode = testmode

		imgs_dict = {Path(img).stem:img for img in iglob(f'{img_dir}/*.jpg')}
		with open(meta_file, 'r') as f: annotations_dict = json.load(f)
		annotations_list = annotations_dict['annotations']
		
		data_dict = {}
		for annotation in annotations_list:
			img_id = str(annotation['image_id'])
			cat_id = annotation['category_id'] - 1 #TODO: make sure that category ids start from 1, not 0 (What is this supposed to mean?)
			boxleft,boxtop,boxwidth,boxheight = annotation['bbox']
			if not img_id in imgs_dict: continue

			imwidth,imheight = PIL.Image.open(imgs_dict[img_id]).size
			box_cenx = boxleft + boxwidth/2.
			box_ceny = boxtop + boxheight/2.
			x,y,w,h = box_cenx/imwidth, box_ceny/imheight, boxwidth/imwidth, boxheight/imheight

			if img_id not in data_dict: data_dict[img_id] = []
			data_dict[img_id].append([cat_id,x,y,w,h])

		self.x, self.y, self.ids = [], [], []
		for img_id, labels in data_dict.items():
			self.x.append(imgs_dict[img_id])
			self.y.append(np.array(labels))
			self.ids.append(img_id)

	def __len__(self): return int(np.ceil(len(self.x)/float(self.batch_size)))

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
		with Pool(self.batch_size) as p: batch_x = p.map(PIL.Image.open,batch_x) #contains PIL images

		x_acc, y_acc = [], {}
		original_img_dims = []
		for x,y in zip(batch_x,batch_y):
			W,H = x.size
			original_img_dims.append((W,H))

			x = x.resize(self.input_res[:2])
			x_aug, y_aug = self.augmenter(x,y)
			x_acc.append(np.array(x_aug))
			y_dict = self.label_encoder(y_aug)

			for dimkey,label in y_dict.items():
				if dimkey not in y_acc: y_acc[dimkey] = []
				y_acc[dimkey].append(label)

		return self.get_batch_test(idx,x_acc,y_acc,original_img_dims) if self.testmode else self.get_batch(x_acc,y_acc)

	def get_batch_test(self, idx, x_acc, y_acc, original_img_dims):
		batch_ids = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]
		return batch_ids, original_img_dims, self.preprocessor(np.array(x_acc)), {dimkey:np.array(gt_tensor) for dimkey,gt_tensor in y_acc.items()}

	def get_batch(self, x_acc, y_acc):
		return self.preprocessor(np.array(x_acc)), {dimkey:np.array(gt_tensor) for dimkey,gt_tensor in y_acc.items()}

'''
#if using pyTorch, equivalent is:
from torch.utils.data import Dataset
class TILSequence(Dataset):
	...
'''
class TILPickle(Sequence):
	def __init__(self, pickle_file, batch_size, augmenter, input_size, label_encoder, preprocessor, testmode=False):

		with open(pickle_file, 'rb') as p: self.ids, self.x, self.y = pickle.load(p)
		self.batch_size = batch_size
		self.augmenter = augmenter
		self.input_res = (*input_size[:2][::-1],input_size[2])
		self.label_encoder = label_encoder
		self.preprocessor = preprocessor
		self.testmode = testmode
		
	def __len__(self):return int(np.ceil(len(self.x) / float(self.batch_size)))
	
	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_ids = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]

		x_acc, y_acc = [], {}
		for x,y in zip(batch_x,batch_y):
			x_aug, y_aug = self.augmenter(x,y)
			if x_aug.size != self.input_res[:2]: x_aug.resize(self.input_res)
			x_acc.append(np.array(x_aug))
			y_dict = self.label_encoder(y_aug)

			for dimkey, label in y_dict.items():
				if dimkey not in y_acc: y_acc[dimkey] = []
				y_acc[dimkey].append( label )

		if self.testmode: return batch_ids, self.preprocessor(np.array(x_acc)), {dimkey:np.array(gt_tensor) for dimkey,gt_tensor in y_acc.items()}
		else: return self.preprocessor(np.array(x_acc)), {dimkey:np.array(gt_tensor) for dimkey,gt_tensor in y_acc.items()}