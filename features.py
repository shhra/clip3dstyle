"""Feature Extraction module.

Copyright 2022, Shailesh Mishra

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""
import torch
import torchvision
import torchvision.transforms as T
from loss import softmax3d


class FeatureExtractor(torch.nn.Module):
    """Extract the features from the ResNet module.

    This class iterates through the resnet module to extract the
    features specified by `style layers`. The idea is you begin
    the traversal from the top of the network into layers, blocks and
    eventually the desired convulation layer. Once it reaches the
    end desired layer, it saves the features into the `feature_list`
    which is returned on the forward pass of the model.
    """

    def __init__(self, model, style_layers):
        """Initialize the feature extraction model."""
        super(FeatureExtractor, self).__init__()
        self.model = model.float()
        self.model.eval()
        self.style_layers = style_layers
        self.layers = {
            "layer1": self.model.layer1,
            "layer2": self.model.layer2,
            "layer3": self.model.layer3,
            "layer4": self.model.layer4,
        }
        self.normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )

    def process_blocks(self, block, layer_id, block_id, feature, feature_list):
        """Extract the conv2 features from each block."""
        original = feature
        feats = block.relu1(block.bn1(block.conv1(feature)))
        feat_conv2 = block.conv2(feats)   # Get the feature here
        feats = block.relu2(block.bn2(feat_conv2))
        feats = block.avgpool(feats)
        feats = block.bn3(block.conv3(feats))
        if block.downsample is not None:
            original = block.downsample(feature)
        feats += original
        if f"layer{layer_id}_{block_id}_conv2" in self.style_layers:
            # before appending the features activate it through the softmax
            feature_list.append(softmax3d(feat_conv2))
        return block.relu3(feats)

    def process_layers(self, layer, layer_id, feature, feature_list):
        """Iterate through each layer in the ResNet."""
        for block_id in range(len(layer)):
            feature = self.process_blocks(
                layer[block_id], layer_id, block_id, feature, feature_list)
        return feature

    def extract_features(self, feature, feature_list):
        """Process the module."""
        for x in range(1, 5):
            feature = self.process_layers(
                self.layers[f"layer{x}"], x, feature, feature_list)

    def get_features(self, features):
        """Compute the features."""
        feature_list = []
        self.extract_features(features, feature_list)
        return feature_list

    def forward(self, image):
        """Forward pass."""
        features = self.normalize(image)
        features = self.model.relu1(self.model.bn1(self.model.conv1(features)))
        features = self.model.relu2(self.model.bn2(self.model.conv2(features)))
        features = self.model.relu3(self.model.bn3(self.model.conv3(features)))
        features = self.model.avgpool(features)
        return self.get_features(features)


class VGGFeatures(torch.nn.Module):
    """Use VGG to extract features."""

    def __init__(self, device):
        """Initialize the model."""
        super(VGGFeatures, self).__init__()
        self.vgg = torchvision.models.vgg16(
            weights='DEFAULT').eval().to(device)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.layers = [11, 13, 15]

    def forward(self, image):
        """Execute the forward pass and feature extraction."""
        feature = self.normalize(image)
        extracted_features = []
        for idx, layer in enumerate(self.vgg.features):
            feature = layer(feature)
            if idx in self.layers:
                extracted_features.append(feature)
            if idx == self.layers[-1]:
                break
        return extracted_features
