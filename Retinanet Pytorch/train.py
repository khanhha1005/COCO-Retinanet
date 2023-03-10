import argparse
import collections

import numpy as np
import time
import torch
import torch.optim as optim
from torchvision import transforms

from Model import model
from Model.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from Model import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')

    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        # dataset_train = CocoDataset(parser.coco_path, set_name='train',
        #                             transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        # dataset_val = CocoDataset(parser.coco_path, set_name='val',
        #                           transform=transforms.Compose([Normalizer(), Resizer()]))
        dataset_test = CocoDataset(parser.coco_path, set_name='val2017',
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=2, collate_fn=collater, batch_sampler=sampler)
    # Using validation set 
    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=2, drop_last=False)
    # dataloader_val = DataLoader(dataset_val, num_workers=2, collate_fn=collater, batch_sampler=sampler_val)


    # Create the model
    retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):
        print("Epoch - {} Started".format(epoch_num))
        st = time.time()

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        et = time.time()
        print("\n Total Time - {}\n".format(int(et - st)))
        scheduler.step(np.mean(epoch_loss))
        #### USING VALIDATION #####
        # print('Start Validation Process')
        # print("Epoch - {} Started".format(epoch_num))
        # st1 = time.time()
        
        # epoch_loss_validation = []

        # for iter_num, data in enumerate(dataloader_val):
                    
        #     with torch.no_grad():
                
        #         # Forward
        #         classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])

        #         # Calculating Loss
        #         classification_loss = classification_loss.mean()
        #         regression_loss = regression_loss.mean()
        #         loss = classification_loss + regression_loss

        #         #Epoch Loss
        #         epoch_loss_validation.append(float(loss))

        #         print(
        #             'Epoch validation: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
        #                 epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss_validation)))

        #         del classification_loss
        #         del regression_loss
            
        # et1 = time.time()
        # print("\n Total Time - {}\n".format(int(et1 - st1)))

        # Save Model after each epoch
        print('Evaluating dataset')
        coco_eval.evaluate_coco(dataset_test, retinanet)
        print('Saving model after one epochs')
        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')


if __name__ == '__main__':
    main()
