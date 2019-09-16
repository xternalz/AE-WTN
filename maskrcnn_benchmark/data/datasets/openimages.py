import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList

class OpenImages(object):
	def __init__(
		self, root, ann_file, size_file, class_file, img_dir, imagelevel, unseen, transforms=None, n_ids=1440000,
	):
		self.root = root
		self.type = "train" if "train" in ann_file else "val"
		self.img_dir = img_dir
		ann = pd.read_csv(os.path.join(root, ann_file))
		self.image_sizes = pd.read_csv(os.path.join(root, size_file + "_sizes.txt"), names=["ImageID", "height", "width"])
		self.image_sizes = self.image_sizes.set_index("ImageID")
		self.image_groups = ann.groupby("ImageID")
		if imagelevel is not None:
			self.imagelevel = True
			imagelevel_ann = pd.read_csv(os.path.join(root, imagelevel))
			imagelevel_ann = imagelevel_ann[imagelevel_ann.Confidence==0]
			self.imagelevel_groups = imagelevel_ann.groupby("ImageID")
		else:
			self.imagelevel = False

		# random sampling of images (class-balanced)
		if self.type == "train":
			self.ids = np.arange(n_ids)
			if os.path.isfile(os.path.join(root, "train_image_ids.npy")):
				self.real_ids = np.load(os.path.join(root, "train_image_ids.npy"))
			else:
				ln_lens = []
				lns = []
				groupby_ln = ann.groupby('LabelName')
				for _, (ln, df) in enumerate(groupby_ln):
					ln_lens.append(len(df.ImageID.unique()))
					lns.append(ln)
				image_lnlens = []
				for _, (imageid, df) in enumerate(self.image_groups):
					image_lnlens.append([imageid, len(df.LabelName.unique())])
				image_lnlens = pd.DataFrame(image_lnlens, columns = ['ImageID', 'lnlen'])
				image_lnlens.set_index('ImageID',inplace=True)
				lns_argsort = np.argsort(ln_lens)
				ln_imageids = dict()
				seen_imageids = []
				for _, i in enumerate(lns_argsort):
					df = groupby_ln.get_group(lns[i])
					df = df[~df.ImageID.isin(seen_imageids)]
					imageids = np.sort(df.ImageID.unique())
					ln_imageids[lns[i]] = imageids
					seen_imageids += imageids.tolist()
					ln_lens = image_lnlens.loc[ln_imageids[lns[i]]].lnlen.values
					ln_imageids[lns[i]] = ln_imageids[lns[i]][np.argsort(ln_lens)[::-1]]
				ln_imageids_keys = list(ln_imageids.keys())
				cls_ind = 0
				cls_img_ind = [0 for i in range(len(ln_imageids_keys))]
				self.real_ids = [-1 for i in range(len(self.ids))]
				for i in self.ids:
					curr_cls = cls_ind%len(ln_imageids_keys)
					cls_ind += 1
					ln = ln_imageids_keys[curr_cls]
					curr_ln_imageids = ln_imageids[ln]
					curr_img = cls_img_ind[curr_cls]%len(curr_ln_imageids)
					cls_img_ind[curr_cls] += 1
					image_id = curr_ln_imageids[curr_img]
					self.real_ids[i] = image_id
				np.save(os.path.join(root, "train_image_ids.npy"), self.real_ids)
		else:
			if self.imagelevel:
				self.real_ids = list(imagelevel_ann.ImageID.unique())
				self.real_ids += list(ann.ImageID.unique())
				self.real_ids = list(np.unique(np.array(self.real_ids)))
			else:
				self.real_ids = list(ann.ImageID.unique())
			self.ids = np.arange(len(self.real_ids))

		# LabelName to int id
		self.classes = pd.read_csv(os.path.join(root, class_file))
		self.clsnet_inds = list(self.classes.clsnet_ind.values)
		self.ln_to_id = dict()
		self.map_class_id_to_class_name = ['__background__']
		for i in range(len(self.classes)):
			self.ln_to_id[self.classes.loc[i].ln] = i + 1
			self.map_class_id_to_class_name.append(self.classes.loc[i].desc)

		self.unseen = unseen

		self.transforms = transforms

	def __len__(self):
		return len(self.ids)

	def __getitem__(self, idx):
		image_id = self.real_ids[idx]

		# load the image as a PIL Image
		image = Image.open(os.path.join(self.root, self.img_dir, str(image_id) + ".jpg")).convert('RGB')

		# boxlist
		boxlist = self.get_groundtruth(idx)

		# transforms
		if self.transforms is not None:
			image, boxlist = self.transforms(image, boxlist)

		# return the image, the boxlist and the idx in your dataset
		return image, boxlist, idx

	def get_img_info(self, idx):
		# get img_height and img_width. This is used if
		# we want to split the batches according to the aspect ratio
		# of the image, as it can be more efficient than loading the
		# image from disk
		img_height, img_width = self.image_sizes.loc[self.real_ids[idx]].values
		return {"height": img_height, "width": img_width}

	def get_groundtruth(self, idx, imagelevel=False, isgroup=False):
		image_id = self.real_ids[idx]

		# image dataframe
		try:
			image_df = self.image_groups.get_group(image_id)

			# boxes
			image_size = np.tile(self.image_sizes.loc[image_id].values,2).astype(np.float32)
			boxes = image_df[['XMin', 'YMin', 'XMax', 'YMax']].values*np.flip(image_size)
			# and labels
			labels = torch.Tensor([self.ln_to_id[ln] for ln in image_df.LabelName.values]).long()

			# create a BoxList from the boxes
			boxlist = BoxList(boxes, tuple(np.flip(self.image_sizes.loc[image_id].values)), mode="xyxy")
			# add the labels to the boxlist
			boxlist.add_field("labels", labels)

			if isgroup:
				isgroup_column = torch.Tensor(image_df.IsGroupOf.values.tolist()).byte()
				boxlist.add_field("isgroup", isgroup_column)
		except:
			boxlist = BoxList(torch.Tensor([[0,0,0,0]]), (100,100), mode="xyxy")
			# add the labels to the boxlist
			boxlist.add_field("labels", torch.Tensor([-100]).long())

			if isgroup:
				boxlist.add_field("isgroup", torch.Tensor([0]).byte())

		if self.imagelevel and imagelevel:
			try:
				imagelevel_lns = np.unique(self.imagelevel_groups.get_group(image_id).LabelName.values)
				imagelevel_classID = [self.ln_to_id[l] for l in imagelevel_lns]
				return boxlist, np.unique(imagelevel_classID)
			except:
				return boxlist, np.array([])
		else:
			return boxlist
