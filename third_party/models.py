import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

# Added by Fabricio 22.04.2025
from torchvision.models import DenseNet121_Weights, ResNet152_Weights


class PretrainedModel(nn.Module):
    """Pretrained model, either from Cadene or TorchVision."""
    def __init__(self):
        super(PretrainedModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError('Subclass of PretrainedModel must implement forward.')

    def fine_tuning_parameters(self, boundary_layers, lrs):
        """Get a list of parameter groups that can be passed to an optimizer.

        Args:
            boundary_layers: List of names for the boundary layers.
            lrs: List of learning rates for each parameter group, from earlier to later layers.

        Returns:
            param_groups: List of dictionaries, one per parameter group.
        """

        def gen_params(start_layer, end_layer):
            saw_start_layer = False
            for name, param in self.named_parameters():
                if end_layer is not None and name == end_layer:
                    # Saw the last layer -> done
                    return
                if start_layer is None or name == start_layer:
                    # Saw the first layer -> Start returning layers
                    saw_start_layer = True

                if saw_start_layer:
                    yield param

        if len(lrs) != boundary_layers + 1:
            raise ValueError('Got {} param groups, but {} learning rates'.format(boundary_layers + 1, len(lrs)))

        # Fine-tune the network's layers from encoder.2 onwards
        boundary_layers = [None] + boundary_layers + [None]
        param_groups = []
        for i in range(len(boundary_layers) - 1):
            start, end = boundary_layers[i:i+2]
            param_groups.append({'params': gen_params(start, end), 'lr': lrs[i]})

        return param_groups


class CadeneModel(PretrainedModel):
    """Models from Cadene's GitHub page of pretrained networks:
        https://github.com/Cadene/pretrained-models.pytorch
    """
    def __init__(self, model_name, tasks, model_args):
        super(CadeneModel, self).__init__()

        self.tasks = tasks
        self.model_uncertainty = model_args.model_uncertainty

        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.pool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.last_linear.in_features
        if self.model_uncertainty:
            num_outputs = 3 * len(tasks)
        else:
            num_outputs = len(tasks)
        self.fc = nn.Linear(num_ftrs, num_outputs)

    def forward(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x


class TorchVisionModel(PretrainedModel):
    """Models from TorchVision's GitHub page of pretrained neural networks:
        https://github.com/pytorch/vision/tree/master/torchvision/models
    """
    def __init__(self, model_fn, tasks, model_args):
        super(TorchVisionModel, self).__init__()

        self.tasks = tasks
        self.model_uncertainty = model_args.model_uncertainty

        # Was before: self.model = model_fn(pretrained=model_args.pretrained)
        if model_args.model == 'DenseNet121':
                self.model = model_fn(weights=DenseNet121_Weights.IMAGENET1K_V1)
        elif model_args.model == 'ResNet152':
                self.model = model_fn(weights=ResNet152_Weights.IMAGENET1K_V1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        if model_args.model == 'DenseNet121':
            num_ftrs = self.model.classifier.in_features
        elif model_args.model == 'ResNet152':
            num_ftrs = self.model.fc.in_features
#         elif model_args.model == 'Inceptionv3':
#             num_ftrs = self.model.fc.in_features

        if self.model_uncertainty:
            num_outputs = 3 * len(tasks)
        else:
            num_outputs = len(tasks)
        self.model.classifier = nn.Linear(num_ftrs, num_outputs)

    def forward(self, x):
        model_name = self.model.__class__.__name__
        if model_name == 'DenseNet':
            x = self.model.features(x)
            x = F.relu(x, inplace=True)
            x = self.pool(x).view(x.size(0), -1)
        elif model_name == 'ResNet':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.pool(x).view(x.size(0), -1)

#         x = self.model.features(x)
#         x = F.relu(x, inplace=True)
#         x = self.pool(x).view(x.size(0), -1)
#         x = self.model.classifier(x)
        x = self.model.classifier(x)
        return x


class DenseNet121(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(DenseNet121, self).__init__(models.densenet121, tasks, model_args)
        self.gradcam_target_layer = 'model_features_norm5'


class DenseNet161(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(DenseNet161, self).__init__(models.densenet161, tasks, model_args)
        self.gradcam_target_layer = 'model_features_norm5'


class DenseNet201(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(DenseNet201, self).__init__(models.densenet201, tasks, model_args)
        self.gradcam_target_layer = 'model.features.norm5'


class ResNet101(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(ResNet101, self).__init__(models.resnet101, tasks, model_args)
        self.gradcam_target_layer = 'model.layer4.2.conv3'


class ResNet152(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(ResNet152, self).__init__(models.resnet152, tasks, model_args)
        self.gradcam_target_layer = 'model.layer4.2.conv3'

class Inceptionv3(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(Inceptionv3, self).__init__(models.inception_v3, tasks, model_args)
        self.gradcam_target_layer = 'model.Mixed_7c.branch_pool.conv'
        
class Inceptionv4(CadeneModel):
    def __init__(self, tasks, model_args):
        super(Inceptionv4, self).__init__('inceptionv4', tasks, model_args)
