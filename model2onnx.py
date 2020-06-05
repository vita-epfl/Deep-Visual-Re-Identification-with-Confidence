import torch
import torchvision
import os
from torchreid.utils.torchtools import count_num_param
import argparse
from torchreid import models

parser = argparse.ArgumentParser(description='Convert Pytorch model to ONNX')

parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
args = parser.parse_args()

print("Initializing model: {}".format(args.arch))
model = models.init_model(name=args.arch, num_classes=10, loss={'htri'}, use_gpu=True)
print("Model size: {:.3f} M".format(count_num_param(model)))


checkpoint = torch.load(args.load_weights)
pretrain_dict = checkpoint['state_dict']
model_dict = model.state_dict()
pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
model_dict.update(pretrain_dict)
model.load_state_dict(model_dict)
print("Loaded pretrained weights from '{}'".format(args.load_weights))

model.eval()
dummy_input = torch.randn(100, 3, args.height, args.width, requires_grad=True)

model(dummy_input)
torch.onnx.export(model, dummy_input, os.path.splitext(args.load_weights)[0]+".onnx")
