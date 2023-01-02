# COCO-Retinanet
## Installation
1) Clone this repo
2) Install required package	
```
apt-get install tk-dev python-tk
pip install pandas
pip install pycocotools
pip install opencv-python
pip install requests
```
3)Get the data from coco, unzip the data 
```
!wget http://images.cocodataset.org/zips/val2017.zip
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
!wget http://images.cocodataset.org/zips/train2017.zip  
!unzip train2017.zip
!rm -rf train2017.zip
!unzip val2017.zip
!rm -rf val2017.zip
!unzip annotations_trainval2017.zip
!rm -rf annotations_trainval2017.zip
```
4) Get all the file into 1 folder coco
## Training
```
python train.py --dataset coco --coco_path ../coco 
```
## Testing 
```
python test.py --coco_path ~/path/to/coco --model_path path/to/model.pt
```
## Weight for the Retina model 
Weight for model(100_epoch-batch_size_2-resnet_50) in the file weight.pt

The weight for model can be loaded using:
```
retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)
retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
```
## Inference
Using the  inference_model.ipynb , to run the detection of the model 
## Some demo of the training and testing also in the  inference_model.ipynb
