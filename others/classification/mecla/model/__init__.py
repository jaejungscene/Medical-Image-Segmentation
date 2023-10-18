import torch.nn as nn
import timm
from .factory import get_model, get_ddp_model

# def get_model(args):
#     if args.model_name == "resnet50":
#         model = timm.create_model(args.model_name, pretrained=True)
#         infeature = model.fc.in_features
#         model.fc =  nn.Linear(infeature, args.num_classes)
#     elif args.model_name == "densenet121":
#         model = timm.create_model(args.model_name, pretrained=True)
#         infeature = model.classifier.in_features
#         model.classifier =  nn.Linear(infeature, args.num_classes)
#     else:
#         raise Exception("not implemented yet")
#     return model

# def get_ddp_model(model, args):
#     model = nn.DataParallel(model).cuda()
#     return model, model