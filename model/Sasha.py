import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.nn import DataParallel
from datetime import datetime
import torch.nn.init as init
import torch.nn as activation 
#from tqdm import tqdm
#import matplotlib.pyplot as plt
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import cosine_similarity

def replace_layers(act, model, old):
    if isinstance(act, str):
        if act != 'Default':
            new = getattr(activation, act)
            _replace_layers(model, old, new())
    else:
        train_act_name = act['name']
        train_act_acts = act['activations']
        acts = _get_acts(train_act_acts)

        new = getattr(activation, train_act_name)
        _replace_layers(model, old, new(activations=acts))


def _replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            _replace_layers(module, old, new)

        if isinstance(module, old):
            setattr(model, n, new)


def _weights_init_cifar(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def _get_acts(train_act_acts):
    acts = []

    for act_name in train_act_acts:
        acts.append(getattr(F, act_name))

    return acts


class LambdaLayerCifar(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayerCifar, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockCifar(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockCifar, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                tup1 = (0, 0, 0, 0, planes // 4, planes // 4)
                def f(x): return F.pad(x[:, :, ::2, ::2], tup1, 'constant', 0)
                self.shortcut = LambdaLayerCifar(f)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) 
        out = self.relu(out)
        return out


class ResNetCifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCifar, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.apply(_weights_init_cifar)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResNet14(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']

        train_act = params['activation']

        self.model = ResNetCifar(BasicBlockCifar, [2, 2, 2])

        self.model.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.fc = nn.Linear(64, out_channels, bias=True)
        replace_layers(train_act, self.model, nn.ReLU)

    def forward(self, x):
        out = self.model(x)
        return out
    

class SimCLR(nn.Module):
    def __init__(self, params, pretrained_weights_path=None):
        super(SimCLR, self).__init__()
        self.backbone = self._init_backbone(params, pretrained_weights_path)
        self.projection_head = self._init_projection_head(params)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        z_normalized = F.normalize(z, dim=1)
        return z_normalized

    def _init_backbone(self, params, pretrained_weights_path):
        backbone = ResNet14(params)
        if pretrained_weights_path:
            # Загрузка предобученных весов, если файл предоставлен
            pretrained_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model_dict = backbone.state_dict()
            # Фильтрация весов, которые существуют и совпадают по размеру
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrained_dict)
            backbone.load_state_dict(model_dict)
        return backbone

    def _init_projection_head(self, params):
        # Инициализация проекционного слоя
        projection_head = nn.Sequential(
            nn.Linear(params['out_channels'], params['projection_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(params['projection_dim'], params['projection_dim']),
            nn.ReLU(inplace=True)
        )
        return projection_head
    
class Vectorization:
    def __init__(self, params, weight_path_resnet14: str, weight_path_simclr: str):
        """
        Инициализация модели и загрузка весов.

        :param params: Параметры для инициализации модели SimCLR.
        :param weight_path_resnet14: Путь к весам ResNet14.
        :param weight_path_simclr: Путь к весам модели SimCLR.
        """
        self.device = torch.device("cpu")
        print(f"Device set to: {self.device}")

        # Инициализация модели и загрузка весов
        self.model = SimCLR(params, weight_path_resnet14).to(self.device)
        
        # Загрузка весов SimCLR
        simclr_state_dict = torch.load(weight_path_simclr, map_location=self.device)
        new_simclr_state_dict = OrderedDict()
        for k, v in simclr_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_simclr_state_dict[name] = v

        self.model.load_state_dict(new_simclr_state_dict, strict=False)
        torch.compile(self.model)
        
        # Определение трансформаций для изображений
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    
    def vectorize_image(self, image_path: str) -> torch.Tensor:
        """
        Векторизация одного изображения.

        :param image_path: Путь к изображению.
        :return: Векторное представление изображения.
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vector = self.model(image)
        return vector
    
    def vectorize_images(self, image_paths: list) -> torch.Tensor:
        """
        Векторизация множества изображений.

        :param image_paths: Список путей к изображениям.
        :return: Объединённое тензорное представление всех изображений.
        """
        vectors = [self.vectorize_image(path) for path in image_paths]
        return torch.cat(vectors)
    
    @staticmethod
    def cosine_distance(vector_1: torch.Tensor, vector_2: torch.Tensor) -> float:
        """
        Вычисление косинусного расстояния между двумя векторами.

        :param vector_1: Первый вектор.
        :param vector_2: Второй вектор.
        :return: Косинусное расстояние между векторами.
        """
        if not isinstance(vector_1, torch.Tensor):
            vector_1 = torch.tensor(vector_1)
        if not isinstance(vector_2, torch.Tensor):
            vector_2 = torch.tensor(vector_2)
            
        vector_1 = vector_1.unsqueeze(0)
        vector_2 = vector_2.unsqueeze(0)

        cosine_similarity = F.cosine_similarity(vector_1, vector_2, dim=1).item()

        return 1 - cosine_similarity  # Возвращаем косинусное расстояние
    
    def average_cosine_similarity(self, vectors: torch.Tensor) -> float:
        """
        Вычисление среднего косинусного сходства между векторами.

        :param vectors: Тензор с векторными представлениями изображений.
        :return: Среднее косинусное сходство между всеми векторами.
        """
        n = vectors.size(0)
        total_similarity = 0.0
        count = 0
        for i in range(n):
            similarities = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[i + 1:], dim=1)
            total_similarity += similarities.sum().item()
            count += similarities.size(0)
        return total_similarity / count if count > 0 else 0.0
    
