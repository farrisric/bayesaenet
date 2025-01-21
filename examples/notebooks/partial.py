from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
import pyro
import pyro.distributions as dist
import tyxe

# Set random seed for reproducibility
pyro.set_rng_seed(42)

# Generate synthetic data
x1 = torch.rand(50, 1) * 0.3 - 1
x2 = torch.rand(50, 1) * 0.5 + 0.5
x = torch.cat([x1, x2])
y = x.mul(4).add(0.8).cos() + 0.1 * torch.randn_like(x)

x_test = torch.linspace(-2, 2, 401).unsqueeze(-1)
y_test = x_test.mul(4).add(0.8).cos()

# Create dataset and dataloader
dataset = data.TensorDataset(x, y)
loader = data.DataLoader(dataset, batch_size=len(x))

# Plot the data and true function
plt.scatter(x.squeeze(), y, label="Data")
plt.plot(x_test.squeeze(), y_test, label="True Function", linestyle="--")
plt.legend()
plt.savefig("data.png")
plt.close()

# Define the heteroskedastic model
class HeteroskedasticNet(nn.Module):
    def __init__(self):
        super(HeteroskedasticNet, self).__init__()
        self.shared_layers = nn.Sequential(nn.Linear(1, 50), nn.Tanh())
        self.mean_head = nn.Linear(50, 1)  # Outputs the mean
        self.variance_head = nn.Linear(50, 1)  # Outputs the log variance

    def forward(self, x):
        shared = self.shared_layers(x)
        mean = self.mean_head(shared)  # Mean of the output distribution
        log_var = self.variance_head(shared)  # Log variance (for numerical stability)
        scale = torch.nn.functional.softplus(log_var)  # Ensure scale is positive
        return torch.cat([mean, scale], dim=-1)  # Concatenate mean and scale

# Create the model
net = HeteroskedasticNet()

# Define the prior and guide
prior = tyxe.priors.IIDPrior(dist.Normal(0, 1))
guide = partial(tyxe.guides.AutoNormal, init_scale=0.01)

# Define the heteroskedastic likelihood
obs_model = tyxe.likelihoods.HeteroskedasticGaussian(len(x), positive_scale=True)

# Wrap the model in a Bayesian neural network
bnn = tyxe.VariationalBNN(net, prior, obs_model, guide)

# Clear Pyro's parameter store and define the optimizer
pyro.clear_param_store()
optim = pyro.optim.Adam({"lr": 1e-3})

# Train the BNN
elbos = []
def callback(bnn, i, e):
    elbos.append(e)

with tyxe.poutine.local_reparameterization():
    bnn.fit(loader, optim, 100000, callback)

# Plot the ELBO
plt.plot(elbos)
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.savefig("elbo.png")
plt.close()


output = bnn.predict(x_test, num_predictions=32)
m, sd = output[:, 0], output[:, 1]

# Plot the results
plt.scatter(x, y, color="black", label="Data")
plt.plot(x_test, y_test, color="black", linestyle="--", label="True Function")
plt.plot(x_test, m.detach(), color="blue", label="Predicted Mean")
for c in range(1, 4):
    plt.fill_between(
        x_test.squeeze(),
        (m - c * sd).squeeze(),
        (m + c * sd).squeeze(),
        alpha=c * 0.1,
        color="blue",
        label=f"Uncertainty (±{c}σ)" if c == 1 else None,
    )
plt.ylim(-2, 2)
plt.legend()
plt.savefig("predictions_with_uncertainty.png")
plt.close()