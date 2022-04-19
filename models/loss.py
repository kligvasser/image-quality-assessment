import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils.core import imresize
from models.vgg import MultiVGGFeaturesExtractor
from models.regressor import DSDRegressor

class PerceptualLoss(nn.Module):
    def __init__(self, features_to_compute=('conv5_4',), criterion=torch.nn.L1Loss()):
        super(PerceptualLoss, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        loss = 0
        for key in inputs_fea.keys():
            loss += self.criterion(inputs_fea[key], targets_fea[key].detach())

        return loss

class StyleLoss(nn.Module):
    def __init__(self, features_to_compute=('relu1_2', 'relu2_1', 'relu3_1'), criterion=torch.nn.L1Loss()):
        super(StyleLoss, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        loss = 0
        for key in inputs_fea.keys():
            inputs_gram = self._gram_matrix(inputs_fea[key])
            with torch.no_grad():
                targets_gram = self._gram_matrix(targets_fea[key]).detach()

            loss += self.criterion(inputs_gram, targets_gram)

        return loss

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

class BufferStyleLoss(nn.Module):
    def __init__(self, features_to_compute, batch_size, criterion=torch.nn.L1Loss(reduction='none')):
        super(BufferStyleLoss, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()
        self.register_buffer('loss', torch.zeros(batch_size, len(features_to_compute)))

    def forward(self, inputs, targets):
        self.loss.zero_()

        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        for i, key in enumerate(inputs_fea.keys()):
            inputs_gram = self._gram_matrix(inputs_fea[key])
            with torch.no_grad():
                targets_gram = self._gram_matrix(targets_fea[key]).detach()
            self.loss[:, i] = self.criterion(inputs_gram, targets_gram).flatten(start_dim=1).mean(dim=1)

        return self.loss

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

class WassersteinLoss(nn.Module):
    def __init__(self, features_to_compute=('relu1_2', 'relu2_1', 'relu3_1'), criterion=torch.nn.MSELoss()):
        super(WassersteinLoss, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        loss = 0.
        for key in inputs_fea.keys():
            loss += self._sliced_wasserstein(inputs_fea[key], targets_fea[key])

        return loss

    def _sliced_wasserstein(self, inputs, targets):
        inputs = inputs.flatten(start_dim=2)
        targets = targets.flatten(start_dim=2)
        sorrted_inputs, _ = torch.sort(inputs, dim=-1)
        sorrted_targets, _ = torch.sort(targets, dim=-1)
        return self.criterion(sorrted_inputs, sorrted_targets.detach())

class DSDLoss(nn.Module):
    def __init__(self, features_to_compute=('relu1_2', 'relu2_1', 'relu3_1'), scales=(0.5,), criterion=torch.nn.L1Loss()):
        super(DSDLoss, self).__init__()
        self.scales = scales
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        loss = 0.
        for scale in self.scales:
            inputs_scaled = imresize(inputs, scale=scale)
            targets_scaled = imresize(targets, scale=scale)

            inputs_style = self._compute_style(inputs, inputs_scaled)
            with torch.no_grad():
                targets_style = self._compute_style(targets, targets_scaled)

            for key in inputs_style.keys():
                loss += self.criterion(inputs_style[key], targets_style[key].detach())

        return loss

    def _compute_style(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        targets_fea = self.features_extractor(targets)

        style = OrderedDict()
        for key in inputs_fea.keys():
            inputs_gram = self._gram_matrix(inputs_fea[key])
            targets_gram = self._gram_matrix(targets_fea[key])
            diff = inputs_gram - targets_gram
            style.update({key: diff})

        return style

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

class NoRefDSDLoss(nn.Module):
    def __init__(self, features_to_compute=('relu1_2', 'relu2_1', 'relu2_2', 'relu3_1'), scale=0.25, batch_size=1, criterion=torch.nn.L1Loss(reduction='none'), regressor_path=''):
        super(NoRefDSDLoss, self).__init__()
        self.features_to_compute = features_to_compute
        self.scale = scale
        self.batch_size = batch_size
        self.criterion = criterion
        self.regressor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights', 'resnet_se_e2000.pt') if regressor_path == '' else regressor_path
        self.regressor = DSDRegressor()
        self.loss = BufferStyleLoss(features_to_compute=self.features_to_compute, batch_size=self.batch_size, criterion=self.criterion)
        
        # loading regressor
        self.regressor.load_state_dict(torch.load(self.regressor_path, map_location='cpu'))
        self.regressor.eval()

    def forward(self, inputs):
        # predict dsd
        with torch.no_grad():
            inputs_resized = imresize(inputs, scale=self.scale).detach()
            preds = self.regressor(inputs_resized).detach()

        # compute dsd
        dsds = self.loss(inputs, inputs_resized)

        # compute loss
        loss = self.criterion(dsds, preds).mean(dim=0)

        return loss.sum()

class ContextualLoss(nn.Module):
    def __init__(self, features_to_compute=('relu2_1',), h=0.5, eps=1e-5):
        super(ContextualLoss, self).__init__()
        self.h = h
        self.eps = eps
        self.extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute, requires_grad=False).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.extractor(inputs)
        with torch.no_grad():
            targets_fea = self.extractor(targets)

        loss = 0
        for key in inputs_fea.keys():
            loss += self._contextual_loss(inputs_fea[key], targets_fea[key])

        return loss

    def _contextual_loss(self, inputs, targets):
        dist = self._cosine_dist(inputs, targets)
        dist_min, _ = torch.min(dist, dim=2, keepdim=True)

        # Eq (2)
        dist_tilde = dist / (dist_min + self.eps)

        # Eq (3)
        w = torch.exp((1 - dist_tilde) / self.h)

        # Eq (4)
        cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)

        # Eq (1)
        cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
        loss = torch.mean(-torch.log(cx + self.eps))

        return loss

    def _cosine_dist(self, x, y):
        # reduce mean
        y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
        x_centered = x - y_mu
        y_centered = y - y_mu

        # L2 normalization
        x_normalized = F.normalize(x_centered, p=2, dim=1)
        y_normalized = F.normalize(y_centered, p=2, dim=1)

        # channel-wise vectorization
        x_normalized = x_normalized.flatten(start_dim=2)  # (N, C, H*W)
        y_normalized = y_normalized.flatten(start_dim=2)  # (N, C, H*W)

        # consine similarity
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

        # convert to distance
        dist = 1 - cosine_sim

        return dist