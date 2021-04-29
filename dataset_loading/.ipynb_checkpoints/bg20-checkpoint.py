import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import cv2

class BristolGorilla2020(Dataset):

    def __init__(self, path, dataset, transform):
        self.images = dataset['file_name']
        self.annotation_files = dataset['annotation_file']
        self.image_paths = dataset['image_path']
        self.annotation_paths = dataset['annotation_path']
        self.annotations = dataset['annotation']
        self.transform = transform
        self.path = path
        
        # Data
        self.image = None
        self.cropped_image = None
        self.image_height = None
        self.image_width = None

    def __len__(self):
        return len(self.images)
    
    def convert_bbox(self, x, y, w, h):
        xmin = (x - (w / 2)) * self.image_width
        xmax = (x + (w / 2)) * self.image_width
        ymin = (y - (h / 2)) * self.image_height
        ymax = (y + (h / 2)) * self.image_height
        return round(xmin), round(ymin), round(xmax), round(ymax) 

    def __getitem__(self,index):
        image_name = self.images.iloc[index]
        image = cv2.imread(self.image_paths.iloc[index])
        
        self.image_height, self.image_width, _ = image.shape
        
        annotation = self.annotations.iloc[index].split()
        class_label = int(annotation[:1][0])
        bbox = annotation[1:]

        x, y, w, h = bbox[0],\
            bbox[1],\
            bbox[2],\
            bbox[3]

        xmin, ymin, xmax, ymax = self.convert_bbox(float(x),
                                                   float(y),
                                                   float(w),
                                                   float(h))
        
        cropped_image = image[ymin:ymax, xmin:xmax,]
        
        if(self.transform):
            cropped_image = self.transform(cropped_image)

        return cropped_image, class_label


