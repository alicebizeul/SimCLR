import torch
import torch.nn as nn
import torchvision

from simclr.modules.resnet_hacks import modify_resnet_model
from simclr.modules.identity import Identity

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projector, n_features, custom, init_clusters=None, classes=None, learn_std=False):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features
        self.custom = custom
        self.learn_std = learn_std
        if self.custom:
            self.conditional_prior= nn.ParameterDict()
            self.conditional_prior["centroids"] = nn.Parameter(init_clusters,requires_grad=True)
            if self.learn_std:self.conditional_prior["std"] = nn.Parameter(torch.ones(init_clusters.shape),requires_grad=False)
            
        self.classes = classes

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = projector

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        if self.custom:
            means = torch.unsqueeze(self.conditional_prior["centroids"][0, :, :], 0).repeat([z_i.shape[0], 1, 1])

            if self.learn_std: std = torch.unsqueeze(self.conditional_prior["std"][0, :, :], 0).repeat([z_i.shape[0], 1, 1])

            print(std.shape,means.shape,x_i.shape,z_i.shape)

            z_i = torch.unsqueeze(z_i, -1).repeat([1, 1, self.classes])
            z_j = torch.unsqueeze(z_j, -1).repeat([1, 1, self.classes])

            # computing gmm
            dist_i = z_i - means
            dist_j = z_j - means
            print(dist_i.shape)
            if not self.learn_std:
                p_z_y_i = -0.5 * torch.sum(dist_i * dist_i, dim=1)

                p_z_y_j = -0.5 * torch.sum(dist_j * dist_j, dim=1)
            else: 
                p_z_y_i = -0.5 * torch.sum(dist_i * (std*dist_i), dim=1) - torch.log(torch.sqrt(torch.prod(self.conditional_prior["std"][0, :, :])))     # each has to be ten

                p_z_y_j = -0.5 * torch.sum(dist_j * (std*dist_j), dim=1) - torch.log(torch.sqrt(torch.prod(self.conditional_prior["std"][0, :, :])))

            p_y_z_i = torch.nn.Softmax(dim=-1)(p_z_y_i)
            p_y_z_j = torch.nn.Softmax(dim=-1)(p_z_y_j)



        else: 
            p_y_z_i, p_y_z_j = None, None
        return p_y_z_i, p_y_z_j, h_i, h_j, z_i, z_j,
