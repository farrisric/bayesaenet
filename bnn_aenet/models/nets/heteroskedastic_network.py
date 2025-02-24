import torch
import torch.nn as nn
from collections import OrderedDict


class NetAtom(nn.Module):
    """
    A neural network for predicting atomic energies and forces, with support for heteroskedastic
    Gaussian likelihood.
    """

    def __init__(self, input_size, hidden_size, species, active_names, alpha, device,
                 e_scaling, e_shift, dropout=0):
        """
        Initialize the network.

        Args:
            input_size (list): List of input sizes for each species.
            hidden_size (list): List of hidden layer sizes for each species.
            species (list): List of species in the dataset.
            active_names (list): List of activation function names for each species.
            alpha (float): Regularization parameter.
            device (torch.device): Device to run the model on (e.g., "cpu" or "cuda").
            e_scaling (float): Scaling factor for energy.
            e_shift (float): Shift factor for energy.
        """
        super(NetAtom, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.species = species
        self.active_names = active_names
        self.alpha = torch.tensor(alpha)
        self.device = device
        self.e_scaling = e_scaling
        self.e_shift = e_shift
        self.dropout = dropout

        # Define activation functions
        self.activations = {
            "linear": nn.Identity(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }

        # Build the network for each species
        
        self.networks = nn.ModuleList([
            self._build_network(input_size, hidden_size, active_names[i])
            for i, (input_size, hidden_size) in enumerate(zip(input_size, hidden_size))
        ])

    def _build_network(self, input_size, hidden_size, active_names):
        """
        Build a neural network for a single species.

        Args:
            input_size (int): Size of the input descriptor.
            hidden_size (list): Sizes of the hidden layers.
            active_names (list): Names of the activation functions for each layer.

        Returns:
            nn.Sequential: A neural network for the species.
        """
        layers = OrderedDict()
        for i, (n_in, n_out) in enumerate(zip([input_size] + hidden_size[:-1], hidden_size)):
            layers[f"Linear_Layer_{i+1}"] = nn.Linear(n_in, n_out)
            if self.dropout > 0:
                layers[f"Dropout_Layer_{i+1}"] = nn.Dropout(p=self.dropout)
                print(f"Dropout layer {i+1} added with p={self.dropout}")
            layers[f"Activation_Layer_{i+1}"] = self.activations[active_names[i]]
        layers["Output_Layer"] = nn.Linear(hidden_size[-1], 2)  # Output mean and log variance
        return nn.Sequential(layers)

    def forward(self, descriptors, logic_reduce):
        """
        Forward pass for heteroskedastic prediction (mean and log variance).

        Args:
            descriptors (list): List of descriptors for each species.
            logic_reduce (list): List of tensors to reorder atomic contributions.

        Returns:
            tuple: Predicted mean and log variance for each structure.
        """
        means, log_vars = [], []
        for i in range(len(self.species)):
            output = self.networks[i](descriptors[i].float())
            means.append(output[:, 0].unsqueeze(-1))
            log_vars.append(torch.nn.functional.softplus(output[:, 1].unsqueeze(-1)))
        
        structure_means = torch.zeros(len(logic_reduce[0]), device=self.device)
        structure_log_vars = torch.zeros(len(logic_reduce[0]), device=self.device)
        for i in range(len(self.species)):
            if len(logic_reduce[i].shape) == 1:
                logic_reduce[i] = logic_reduce[i].unsqueeze(0)

            structure_means += torch.einsum("ij,ki->k", means[i], logic_reduce[i])
            structure_log_vars += torch.einsum("ij,ki->k", log_vars[i], logic_reduce[i])
        # structure_log_vars = torch.nn.functional.softplus(structure_log_vars)
        return torch.cat([structure_means, structure_log_vars], dim=-1)

    def get_loss_heteroskedastic(self, descriptors, energies, logic_reduce):
        """
        Compute the negative log likelihood under a heteroskedastic Gaussian distribution.

        Args:
            descriptors (list): List of descriptors for each species.
            energies (torch.Tensor): Target energies for each structure.
            logic_reduce (list): List of tensors to reorder atomic contributions.

        Returns:
            torch.Tensor: Negative log likelihood loss.
        """
        means, log_vars = self.forward_heteroskedastic(descriptors, logic_reduce)
        sigma = torch.exp(0.5 * log_vars)  # Transform log variance to standard deviation
        dist = torch.distributions.Normal(means, sigma)
        log_prob = dist.log_prob(energies)  # Compute log probability
        return -log_prob.mean()  # Negative log likelihood

    def save(self, path: str) -> None:
        """
        Save the model state to a file.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=torch.device("cpu")):
        """
        Load the model state from a file.

        Args:
            path (str): Path to load the model from.
            map_location (torch.device): Device to load the model onto.
        """
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
