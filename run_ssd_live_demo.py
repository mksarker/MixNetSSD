from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.ego_images import OpenImagesDataset
# from vision.datasets.ub_ego_images import OpenImagesDataset
import argparse
from vision.utils.misc import Timer
import cv2
import sys
import torch

# parser = argparse.ArgumentParser(description="SSD Evaluation on EGO Dataset.")
# parser.add_argument("--dataset_type", default="ego", type=str,
#                     help='Specify dataset type. Currently support voc and open_images.')

# args = parser.parse_args()


dataset = OpenImagesDataset(root='data/ego', dataset_type="val")

if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <net type>  <model path> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

# class_names=[]
with open(label_path, 'r') as f:
    class_names = [name.strip() for name in f.readlines()]

num_classes = len(class_names)

print(class_names)

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)


import moviepy.editor as mpy

# fourcc = cv2.VideoWriter_fourcc(*'XVID') # cv2.cv.CV_FOURCC(*'X264')
# out = cv2.VideoWriter('food_detection.avi',fourcc, 1, (480,270))

rendered_frames = []
font = cv2.FONT_HERSHEY_DUPLEX
color=(0, 255, 255 )

timer = Timer()
for i in range(500,1000): #len(dataset)
    orig_image =  dataset.get_image(i)
    # orig_image= cv2.resize(orig_image,(300, 300))
    # height, width = orig_image.shape
    if orig_image is None:
        continue
    # image = orig_image
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

        cv2.putText(orig_image, label,(box[0]+0, box[1]+10), cv2.FONT_HERSHEY_COMPLEX,0.4,(255, 255, 0),1,cv2.LINE_AA) 
        # cv2.putText(orig_image,label,(1, 20 * (i + 2) + 4),font,0.5,color,1,cv2.LINE_AA)
    # cv2.imshow('Food Detection', orig_image)
        rendered_frames.append(orig_image)
    # out.write(orig_image)
    # if cv2.waitKey(500) & 0xFF == ord('q'):
        # break

clip = mpy.ImageSequenceClip(rendered_frames, fps=2)
clip.write_videofile('food_detection.mp4')

cap.release()
cv2.destroyAllWindows()
