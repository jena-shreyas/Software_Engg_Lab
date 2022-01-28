#Imports
import json
import numpy as np
import cv2
from my_package.data.transforms.blur import BlurImage
from my_package.data.transforms.crop import CropImage
from my_package.data.transforms.rescale import RescaleImage
from my_package.data.transforms.rotate import RotateImage
from my_package.data.transforms.flip import FlipImage

class Dataset(object):
    '''
        A class for the dataset that will return data items as per the given index
    '''

    def __init__(self, annotation_file, transforms = None):
        '''
            Arguments:
            annotation_file: path to the annotation file
            transforms: list of transforms (class instances)
                        For instance, [<class 'RandomCrop'>, <class 'Rotate'>]
        '''
        self.annotation_file=annotation_file
        self.transforms=transforms
        self.json_list=[]

        with open(annotation_file,'r') as json_file:
            self.json_list=list(json_file)

    def __len__(self):
        '''
            return the number of data points in the dataset
        '''
        return len(self.json_list)

    def __getitem__(self, idx):
        '''
            return the dataset element for the index: "idx"
            Arguments:
                idx: index of the data element.

            Returns: A dictionary with:
                image: image (in the form of a numpy array) (shape: (3, H, W))
                gt_png_ann: the segmentation annotation image (in the form of a numpy array) (shape: (1, H, W))
                gt_bboxes: N X 5 array where N is the number of bounding boxes, each 
                            consisting of [class, x1, y1, x2, y2]
                            x1 and x2 lie between 0 and width of the image,
                            y1 and y2 lie between 0 and height of the image.

            You need to do the following, 
            1. Extract the correct annotation using the idx provided.
            2. Read the image, png segmentation and convert it into a numpy array (wont be necessary
                with some libraries). The shape of the arrays would be (3, H, W) and (1, H, W), respectively.
            3. Scale the values in the arrays to be with [0, 1].
            4. Perform the desired transformations on the image.
            5. Return the dictionary of the transformed image and annotations as specified.
        '''
       
        annot_str=[data for i,data in enumerate(self.json_list) if i==idx][0]
        annot=json.loads(annot_str)

        image=cv2.imread('./data/'+annot["img_fn"])
        gt_png_ann=cv2.imread('./data/'+annot["png_ann_fn"],cv2.IMREAD_GRAYSCALE)

        for transform in self.transforms:
            image=transform(image)

        image=image.transpose((2,0,1))/255
        png_shape=gt_png_ann.shape
        gt_png_ann=np.reshape(gt_png_ann,(-1,png_shape[0],png_shape[1]))/255
        gt_bboxes=[[0.0 for i in range(5)] for j in range(len(annot['bboxes']))]

        for i,bbox in enumerate(annot['bboxes']):

            gt_bboxes[i][0]=bbox['category']
            gt_bboxes[i][1]=bbox['bbox'][0]
            gt_bboxes[i][2]=bbox['bbox'][1]
            gt_bboxes[i][3]=bbox['bbox'][2]
            gt_bboxes[i][4]=bbox['bbox'][3]

        dict={'image':image, 'gt_png_ann':gt_png_ann, 'gt_bboxes':gt_bboxes}
        return dict

'''
blur=BlurImage(5)
crop=CropImage((250,200),'random')
flip=FlipImage('horizontal')
rescale=RescaleImage((429,640))
rotate=RotateImage(45)

transforms=[blur,crop,flip,rescale,rotate]
data=Dataset("data/annotations.jsonl",transforms)   
dict=data[1]
cv2.namedWindow('Final image',cv2.WINDOW_FULLSCREEN)
img=dict['image']
img=np.stack((img[0,:,:],img[1,:,:],img[2,:,:]),axis=2)
print(img.shape)
img=np.array(img*255,dtype=np.uint8)
cv2.imshow('Final image',img)
k=cv2.waitKey(0)

if k==ord('q'):
    cv2.destroyAllWindows()
'''