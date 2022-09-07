import gc
import torch
import numpy as np
from sklearn.utils.validation import check_is_fitted

from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

class GPLVMModel(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing):
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)
    
        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
    
        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
    
        # Initialise X with randn
        X_init = torch.nn.Parameter(torch.randn(n, latent_dim))
          
        # LatentVariable (c)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)
        
        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)
        
        super().__init__(X, q_f)
        
        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)


class GPLVM:
    def __init__(self, n, data_dim, latent_dim, n_inducing, batch_size, epochs,
        patience, n_neurons, dropout, activation, verbose):
        self.n_inducing = 500
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.history = []

        # Model
        self.model = GPLVMModel(n, data_dim, latent_dim, n_inducing)

        # Likelihood
        self.likelihood = GaussianLikelihood(batch_shape=self.model.batch_shape)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

        # Declaring the objective to be optimised along with optimiser 
        # (see models/latent_variable.py for how the additional loss terms are accounted for)
        self.mll = VariationalELBO(self.likelihood, self.model, num_data=n)
            
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()}
        ], lr=0.01)

    def _fit(self, X):
        loss_list = []
        for i in range(self.epochs): 
            batch_index = self.model._get_batch_idx(self.batch_size)
            self.optimizer.zero_grad()
            sample = self.model.sample_latent_variable()  # a full sample returns latent x across all n
            sample_batch = sample[batch_index]
            output_batch = self.model(sample_batch)
            loss = -self.mll(output_batch, X[batch_index].T).sum()
            loss_list.append(loss.item())
            if self.verbose:
                print(f'epoch {i}: loss: {str(float(np.round(loss.item(), 2)))}')
            loss.backward()
            self.optimizer.step()
        self.history = loss_list
        gc.collect()
        torch.cuda.empty_cache()

    def fit(self, X):
        self._fit(X)
        return self
    
    def _transform(self, X):
        X_transformed = self.model.sample_latent_variable()
        X_transformed = X_transformed[np.arange(X.shape[0])]
        return X_transformed

    def transform(self, X):
        check_is_fitted(self, 'history')
        return self._transform(X)

    def fit_transform(self, X):
        self._fit(X)
        return self._transform(X)
    
    def reconstruct(self, X_transformed):
        X_rec_list = []
        steps = X_transformed.shape[0] // self.batch_size
        for i in range(steps):
            X_reconstructed_batch = self.model(X_transformed[(i * self.batch_size) : (i + 1) * self.batch_size, :]).mean.T.detach().cpu().numpy()
            X_rec_list.append(X_reconstructed_batch)
        X_reconstructed = np.concatenate(X_rec_list, axis=0)
        return X_reconstructed

    def get_history(self):
        check_is_fitted(self, 'history')
        return self.history
