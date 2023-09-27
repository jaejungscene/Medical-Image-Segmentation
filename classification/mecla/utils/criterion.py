import torch
from torch import nn
from torch.nn import functional as F, BCEWithLogitsLoss, MSELoss
from torch.cuda.amp import GradScaler


class NativeScalerWithGradAccum:
    def __init__(self):
        """NativeScalerWithGradAccum (timm)
        Native(pytorch) f16 scaler
        """
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, model_param, scheduler=None, grad_norm=None, update=True):
        self._scaler.scale(loss).backward()
        if update:
            if grad_norm:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_param, grad_norm)
            self._scaler.step(optimizer)
            self._scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class BCEWithLogitsLossWithTypeCasting(BCEWithLogitsLoss):
    def forward(self, y_hat, y):
        y = y.float()
        y = y.reshape(y_hat.shape)
        return super().forward(y_hat, y)


def get_criterion_scaler(args):
    """Get Criterion(Loss) function and scaler
    Criterion functions are divided depending on usage of mixup
    - w/ mixup - you don't need to add smoothing loss, because mixup will add smoothing loss.
    - w/o mixup - you should need to add smoothing loss
    """
    if args.criterion in ['ce', 'crossentropy']:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    elif args.criterion in ['bce', 'binarycrossentropy']:
        criterion = BCEWithLogitsLossWithTypeCasting()
    elif args.criterion in ['mse', 'l2']:
        criterion = MSELoss()

    if args.amp:
        scaler = NativeScalerWithGradAccum()
    else:
        scaler = None

    return criterion, scaler