#Imports
from turtle import width
import numpy as np
import cv2

def plot_visualization(image,bboxes,output): # Write the required arguments

    # The function should plot the predicted segmentation maps and the bounding boxes on the images and save them.
    # Tip: keep the dimensions of the output image less than 800 to avoid RAM crashes.
    image=np.uint8(image.transpose((1,2,0))*255)
    for i in range(min(3,len(bboxes))):

        top_left=(bboxes[i][1],bboxes[i][2])
        bottom_right=(bboxes[i][3],bboxes[i][4])
        green=(0,255,0)
        cv2.rectangle(image,top_left,bottom_right,green,width=5)
        cv2.putText(image,bboxes[i][0],top_left,fill='green')

    cv2.imwrite(output,image)


    