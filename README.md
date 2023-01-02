# COCO-Retinanet
## Training
```
python train.py --dataset coco --coco_path ../coco 
```
## Testing 
```
python coco_validation.py --coco_path ~/path/to/coco --model_path path/to/model.pt
```
## Weight for the Retina model 

- https://drive.google.com/open?id=1yLmjq3JtXi841yXWBxst0coAgR26MNBS 


The state dict model can be loaded using:
```
retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)
retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
```
## Inference
Using the  inference_model.ipynb , to run the detection of the model 
