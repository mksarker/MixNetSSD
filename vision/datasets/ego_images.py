import numpy as np
import pathlib
import cv2
import pandas as pd

from PIL import Image


class OpenImagesDataset:

    def __init__(self, root, dataset_type='train', transform=None, target_transform=None, balance_data=False):
                 
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])  
        boxes = image_info['boxes']
        boxes[:, 0] *= image.shape[1]
        boxes[:, 1] *= image.shape[0]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 3] *= image.shape[0]
        labels = image_info['labels']
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)

        return image, boxes, labels

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _read_data(self):
        annotation_file = f"{self.root}/{self.dataset_type}.csv"
        annotations = pd.read_csv(annotation_file)
        class_names = ['BACKGROUND'] + sorted(list(annotations['class_ids'].unique()))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []
        for image_id, group in annotations.groupby("image"):
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            # print( type(boxes))
            labels = np.array([class_dict[name] for name in group["class_ids"]])
            data.append({
                'image_id': image_id,
                'boxes': boxes,
                'labels': labels
            })
        return data, class_names, class_dict
       

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)

    def _read_image(self, image_id):
        image_file = image_id
        # print(image_id)
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
    
    # def _read_image(self, image_id):
    #     image_file = image_id
    #     print(image_file)
    #     image = cv2.imread(str(image_file))
    #     # image=cv2.resize(image,(10,10))
    #     print(image.shape)
    #     # print('#################')
    #     if image.shape[2] == 1:
    #         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #     else:
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     return image

    def _balance_data(self):
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data

# from torch.utils.data import DataLoader, ConcatDataset

# # import pandas as pd 

# # data= pd.read_csv('/media/mostafa/RESEARCH/EPIC_KITCHENS_2018/detection_code/pytorch-ssd/data/ego/EPIC_noun_classes.csv') 
# # labels=list(data['class_key'])

# # Define model
# import torch.nn as nn
# import torch.nn.functional as F


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()

# import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# if __name__ == '__main__':
#     dataset_path='C:/Users/3055638/Documents/code/MixNet_SSD/data/ego/'
#     dataset = OpenImagesDataset(dataset_path, dataset_type='val')

#     # trainloader = DataLoader(dataset, batch_size=2)

#     # print(len(trainloader))

#     # for epoch in range(2):  # loop over the dataset multiple times

#         # running_loss = 0.0
#         # for i, data in enumerate(trainloader):
#         #     print(i)
#         #     # get the inputs
#         #     inputs, boxes, labels = data

#         #     # zero the parameter gradients
#         #     optimizer.zero_grad()

#         #     # forward + backward + optimize
#         #     outputs = net(inputs)
#         #     loss = criterion(outputs, labels)
#         #     loss.backward()
#         #     optimizer.step()

#         #     # print statistics
#         #     running_loss += loss.item()
#         #     if i % 2000 == 1999:    # print every 2000 mini-batches
#         #         print('[%d, %5d] loss: %.3f' %
#         #             (epoch + 1, i + 1, running_loss / 2000))
#         #         running_loss = 0.0

#         # print('Finished Training')



#     print(dataset)

#     num_classes = len(dataset.class_names)
#     print(num_classes)


#     image, boxes, lbl= dataset. __getitem__(0)
#     print ('boxes:', boxes)
#     print('labels:', lbl)

#     image=dataset.get_image(0)
    
#     x1= boxes[:, 0] 
#     y1=boxes[:, 1]
#     x2=boxes[:, 2] 
#     y2= boxes[:, 3]
    
#     # cv2.resize(image,(512,300))
#     cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),1)
#     cv2.putText(image,'cd8' ,(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2,cv2.LINE_AA)  

#     cv2.imshow('my image',image)
#     cv2.waitKey(0)

    

    # train_loader = DataLoader(dataset, batch_size=2)



