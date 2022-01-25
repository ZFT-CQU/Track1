import SimpleITK
import numpy as np

from pandas import DataFrame
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import json
from typing import Dict
import utils.utils as utils
from utils.dataset import CXRNoduleDataset, get_transform
import os
from utils.train import train_one_epoch, validate_one_epoch
import itertools
from pathlib import Path
from postprocessing import get_NonMaxSup_boxes
from torch.optim import lr_scheduler
from lib.resnet import ResNet_FPN

# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
execute_in_docker = True


class Noduledetection(DetectionAlgorithm):
    def __init__(self, input_dir, output_dir, train=True):
        super().__init__(validators=dict(input_image=(
            UniqueImagesValidator(),
            UniquePathIndicesValidator(),
        )),
            input_path=Path(input_dir),
            output_file=Path(
            os.path.join(output_dir, 'nodules.json')))

        # ------------------------------- LOAD the model here ---------------------------------
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.input_path, self.output_path = input_dir, output_dir
        print('using the device ', self.device)

        if train:
            backbone = ResNet_FPN(backbone_name='resnext101_32x8d', pretrained=True)
        else:
            backbone = ResNet_FPN(backbone_name='resnext101_32x8d', pretrained=False)

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        self.model = FasterRCNN(backbone,
                                num_classes=2,
                                rpn_anchor_generator=anchor_generator,
                                box_roi_pool=roi_pooler)

        if not train:
            # retrain or test phase
            print('loading the model.pth file :')
            self.model.load_state_dict(
                torch.load(
                    Path("/opt/algorithm/model.pth")
                    if execute_in_docker else Path("model.pth"),
                    map_location=self.device,
                ))

        self.model.to(self.device)

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    # TODO: Copy this function for your processor as well!
    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Detect and score candidates
        scored_candidates = self.predict(input_image=input_image)

        # Write resulting candidates to nodules.json for this case
        return scored_candidates

    # --------------------Write your retrain function here ------------

    def train(self, num_epochs=1, batch_size=8):
        '''
        input_dir: Input directory containing all the images to train with
        output_dir: output_dir to write model to.
        num_epochs: Number of epochs for training the algorithm.
        '''
        # Implementation of the pytorch model and training functions is based on pytorch tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

        # create training dataset and defined transformations
        self.model.train()
        input_dir = self.input_path
        train_set = CXRNoduleDataset(input_dir,
                                     os.path.join(input_dir, 'metadata.csv'),
                                     get_transform(train=True),
                                     phase='train')
        val_set = CXRNoduleDataset(input_dir,
                                   os.path.join(input_dir, 'metadata.csv'),
                                   get_transform(train=True),
                                   phase='val')
        # define training and validation data loaders
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=16,
                                                   collate_fn=utils.collate_fn)
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=16,
                                                 collate_fn=utils.collate_fn)
        print('training starts ')

        # construct an optimizer
        parameters = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(params=parameters, lr=1e-4, weight_decay=1e-4)  # true wd, filter_bias_and_bn
        self.ema_model = utils.ModelEma(self.model, 0.9997)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=num_epochs,
                                            pct_start=0.2)

        for epoch in range(num_epochs):
            print('#')
            print('#')
            print('#')
            print('#')
            train_one_epoch(self.model, self.ema_model, optimizer, scheduler, train_loader, self.device,
                            epoch)
            # evaluate regular model
            auc_train_regular = validate_one_epoch(self.model,
                                                   train_loader,
                                                   self.device,
                                                   phase='Train',
                                                   epoch=epoch,
                                                   model_name='regular')
            auc_val_regular = validate_one_epoch(self.model,
                                                 val_loader,
                                                 self.device,
                                                 phase='Val',
                                                 epoch=epoch,
                                                 model_name='regular')
            # evaluate ema_model
            auc_train_ema = validate_one_epoch(self.ema_model.module,
                                               train_loader,
                                               self.device,
                                               phase='Train',
                                               epoch=epoch,
                                               model_name='ema')
            auc_val_ema = validate_one_epoch(self.ema_model.module,
                                             val_loader,
                                             self.device,
                                             phase='Val',
                                             epoch=epoch,
                                             model_name='ema')

            # update the learning rate
            scheduler.step()
            # evaluate on the test dataset
            # IMPORTANT: save retrained version frequently.
            regular_model_path = os.path.join(
                self.output_path, 'epoch_{}_regular_model_{:.4f}_{:.4f}.pth'.format(
                    epoch, auc_train_regular, auc_val_regular))
            ema_model_path = os.path.join(
                self.output_path, 'epoch_{}_ema_model_{:.4f}_{:.4f}.pth'.format(
                    epoch, auc_train_ema, auc_val_ema))
            print('save regular model to ' + regular_model_path)
            print('save ema model to ' + ema_model_path)
            torch.save(self.model.state_dict(), regular_model_path)
            torch.save(self.ema_model.module.state_dict(), ema_model_path)

    def format_to_GC(self, np_prediction, spacing) -> Dict:
        '''
        Convenient function returns detection prediction in required grand-challenge format.
        See:
        https://comic.github.io/grandchallenge.org/components.html#grandchallenge.components.models.InterfaceKind.interface_type_annotation


        np_prediction: dictionary with keys boxes and scores.
        np_prediction[boxes] holds coordinates in the format as x1,y1,x2,y2
        spacing :  pixel spacing for x and y coordinates.

        return:
        a Dict in line with grand-challenge.org format.
        '''
        # For the test set, we expect the coordinates in millimeters.
        # this transformation ensures that the pixel coordinates are transformed to mm.
        # and boxes coordinates saved according to grand challenge ordering.
        x_y_spacing = [spacing[0], spacing[1], spacing[0], spacing[1]]
        boxes = []
        for i, bb in enumerate(np_prediction['boxes']):
            box = {}
            box['corners'] = []
            x_min, y_min, x_max, y_max = bb * x_y_spacing
            x_min, y_min, x_max, y_max = round(x_min,
                                               2), round(y_min, 2), round(
                                                   x_max, 2), round(y_max, 2)
            bottom_left = [x_min, y_min, np_prediction['slice'][i]]
            bottom_right = [x_max, y_min, np_prediction['slice'][i]]
            top_left = [x_min, y_max, np_prediction['slice'][i]]
            top_right = [x_max, y_max, np_prediction['slice'][i]]
            box['corners'].extend(
                [top_right, top_left, bottom_left, bottom_right])
            box['probability'] = round(float(np_prediction['scores'][i]), 2)
            boxes.append(box)

        return dict(type="Multiple 2D bounding boxes",
                    boxes=boxes,
                    version={
                        "major": 1,
                        "minor": 0
                    })

    def merge_dict(self, results):
        merged_d = {}
        for k in results[0].keys():
            merged_d[k] = list(itertools.chain(*[d[k] for d in results]))
        return merged_d

    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        self.model.eval()

        image_data = SimpleITK.GetArrayFromImage(input_image)
        spacing = input_image.GetSpacing()
        image_data = np.array(image_data)

        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, 0)

        results = []
        # operate on 3D image (CXRs are stacked together)
        for j in range(len(image_data)):
            # Pre-process the image
            image = image_data[j, :, :]
            # The range should be from 0 to 1.
            image = image.astype(np.float32) / np.max(image)  # normalize
            image = np.expand_dims(image, axis=0)
            tensor_image = torch.from_numpy(image).to(
                self.device)  # .reshape(1, 1024, 1024)
            with torch.no_grad():
                prediction = self.model([tensor_image.to(self.device)])

            prediction = [get_NonMaxSup_boxes(prediction[0])]
            # convert predictions from tensor to numpy array.
            np_prediction = {
                str(key): [i.cpu().numpy() for i in val]
                for key, val in prediction[0].items()
            }
            np_prediction['slice'] = len(np_prediction['boxes']) * [j]
            results.append(np_prediction)

        predictions = self.merge_dict(results)
        data = self.format_to_GC(predictions, spacing)
        print(data)
        return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='process.py',
        description='Reads all images from an input directory and produces '
        'results in an output directory')

    parser.add_argument('input_dir', help="input directory to process")
    parser.add_argument('output_dir',
                        help="output directory generate result files in")
    parser.add_argument('--train',
                        action='store_true',
                        help="Algorithm on train mode.")
    parser.add_argument('--retrain',
                        action='store_true',
                        help="Algorithm on retrain mode (loading previous weights).")
    parser.add_argument('--retest',
                        action='store_true',
                        help="Algorithm on evaluate mode after retraining.")
    parser.add_argument('--test',
                        action='store_true',
                        help="Algorithm on evaluate mode after retraining.")
    parser.add_argument('--batch-size',
                        default=8,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 16)')
    parser.add_argument('--epochs',
                        default=50,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    if args.train:  # train mode
        Noduledetection(args.input_dir, args.output_dir,
                        train=True).train(num_epochs=args.epochs,
                                          batch_size=args.batch_size)
    else:  # test mode
        Noduledetection(args.input_dir, args.output_dir, train=False).process()
