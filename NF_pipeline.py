# install requirements: icarogw 2.0.3 + zuko

from icarogw.cupy_pal import get_module_array
from icarogw.conversions import detector2source, detector2source_jacobian

import numpy as np
import torch
from zuko.flows import NSF
from sklearn.model_selection import train_test_split

import copy
import random

import matplotlib.pyplot as plt


class one_catalog_wrapper(object):
    """
    Wrapper to train a Normalizing Flow model on a single BBH synthesis population catalog using the Zuko Neural Spline Flows.

    Parameters
    ----------
    dicto_pop: dict
             Dictionary containing the BBH population samples in log space ('log_mass1', 'log_mass2', 'log_redshift') 
             Note: the redshift samples are drawn from R(z)*dVc/dz/(1+z) where R(z) is the merger rate in unit of events per year and Gpc^3
    global_dicto_pop: dict
             Dictionary containing the global BBH population samples, used to compute the min and max bounds for input normalization
    n_layers: int
             Number of transform layers 
    hidden_features: int
             Number of hidden neurons  
    bins: int
             Number of bins for the spline transformation
    n_iter: int
             Maximum number of training iterations 
    batch_size: int
             Number of samples per training batch
    lr: float
             Learning rate for the Adam optimizer
    patience: int
             Number of iterations to wait for validation loss improvement before early stopping
    seed: int
             Random seed for reproducibility 
    b: float
             Bound for the data normalization. Data is scaled to match the Zuko domain [-b, b]
    plot: bool, optional
             If True, plots the training and validation loss curves (default is False)
    diagnostics: bool, optional
             If True, computes and prints the best training and validation losses (default is False)
    """    
    def __init__(self, dicto_pop, n_layers, hidden_features, bins, n_iter, batch_size, lr, patience, seed, b, plot=False, diagnostics=False):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.population_parameters = None
        self.b = b
        
        raw_pop = np.vstack([
            dicto_pop['log_mass1'].flatten(),
            dicto_pop['log_mass2'].flatten(),
            dicto_pop['log_redshift'].flatten()
        ]).T

        self.min_val = raw_pop.min(axis=0)
        self.max_val = raw_pop.max(axis=0)

        train_raw, vali_raw = train_test_split(raw_pop, test_size=0.2, random_state=seed)

        self.train_data = 2*self.b * (train_raw - self.min_val) / (self.max_val - self.min_val) - self.b
        self.vali_data  = 2*self.b * (vali_raw  - self.min_val) / (self.max_val - self.min_val) - self.b

        np.random.shuffle(self.train_data)

        self.plot = plot
        self.diagnostics = diagnostics

        self.flow = self.build_flow(n_layers=n_layers, hidden_features=hidden_features, bins=bins)

        self.train_flow(self.flow, self.train_data, self.vali_data, n_iter=n_iter, lr=lr, batch_size=batch_size, patience=patience )

    def build_flow(self, n_layers, hidden_features, bins):
        """
        This method constructs the architecture with 3 features: log_mass_1, log_mass_2, log_z.
        """
        return NSF(features=3, context=0, transforms=n_layers, hidden_features=[hidden_features], bins=bins, randperm=True)

    def train_flow(self, flow, train_data, vali_data, n_iter, lr, batch_size, patience):
        """
        This method trains the Normalizing Flow using the Adam optimizer and a learning rate scheduler.
        It implements early stopping based on the validation loss to prevent overfitting.
        """
        optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10 )

        flow.train()

        max_vali_loss = 1e10
        min_vali_model = None
        p = 0
        train_losses = []
        vali_losses = []

        self.N_vali = len(vali_data)

        for i in range(n_iter):
            idx = np.random.choice(len(train_data), size=batch_size, replace=False)
            x_batch = torch.tensor(train_data[idx], dtype=torch.float32)
            optimizer.zero_grad()
            loss = -flow().log_prob(x_batch).mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                vali_tensor = torch.tensor(vali_data, dtype=torch.float32)
                vali_loss = -flow().log_prob(vali_tensor).mean().item()

            train_losses.append(loss.item())
            vali_losses.append(vali_loss)

            scheduler.step(vali_loss)
          
            if vali_loss < max_vali_loss - 1e-4:
                max_vali_loss = vali_loss
                self.best_vali_loss = max_vali_loss
                min_vali_model = copy.deepcopy(flow.state_dict())
                p = 0
            else:
                p += 1

            if p >= patience:
                break

        flow.load_state_dict(min_vali_model)
        flow.eval()
        
        if self.diagnostics == True:
               flow.load_state_dict(min_vali_model)
        with torch.no_grad():
            train_tensor = torch.tensor(train_data, dtype=torch.float32)
            best_train_loss = -flow().log_prob(train_tensor).mean().item()
            self.best_train_loss = best_train_loss
            #print(f"Best training loss: {self.best_train_loss:.4f}")
            #print(f"Best validation loss: {self.best_vali_loss:.4f}")

          
        if self.plot:
            plt.figure(figsize=(6, 3))
            plt.plot(train_losses, label='Training loss')
            plt.plot(vali_losses, label='Validation loss')
            plt.title("Training and Validation losses")
            plt.xlabel("i")
            plt.ylabel("Negative Log Likelihood")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

    def log_pdf(self, log_mass_1, log_mass_2, log_z):
        """
        This method returns the natural logarithm of the pdf of the trained flow at mass1, mass2, z values.
        It handles the [-b, b] normalization of the input data and applies the necessary log-determinant 
        Jacobian correction for the scaling transformation.
        """        
        or_shape = log_mass_1.shape
        data = np.vstack([
            log_mass_1.flatten(),
            log_mass_2.flatten(),
            log_z.flatten()]).T

        norm_data = 2*self.b * (data - self.min_val) / (self.max_val - self.min_val) - self.b
        data_tensor = torch.tensor(norm_data, dtype=torch.float32)

        with torch.no_grad():
            log_probs = self.flow().log_prob(data_tensor).numpy()

        log_det_jac = np.sum(np.log(2*self.b / (self.max_val - self.min_val)))
        log_probs += log_det_jac

        log_probs = np.clip(log_probs, -1e30, 1e30)
        return np.reshape(log_probs, or_shape)


    def sample(self, Nsamples, log_mmin, log_mmax, log_zmin, log_zmax):
        """
        This method draws samples from the trained flow using weighted resampling with a uniform prior.
        """
        log_m1_samples = np.random.uniform(log_mmin, log_mmax, size=Nsamples * 100)
        log_m2_samples = np.random.uniform(log_mmin, log_mmax, size=Nsamples * 100)
        log_z_samples = np.random.uniform(log_zmin, log_zmax, size=Nsamples * 100)

        logw = self.log_pdf(log_m1_samples, log_m2_samples, log_z_samples)
        logw -= logw.max()
        w = np.exp(logw)

        idx = np.random.choice(len(w), size=Nsamples, replace=True, p=w / w.sum())

        return log_m1_samples[idx], log_m2_samples[idx], log_z_samples[idx]

    r"""
    This is a rate model that parametrizes the BBH rate per year at the detector in terms of source-frame
    masses and redshift rate evolution times differential of comoving volume. 
    The source-frame masses and merger redshift distribution follow BBH sythesis catalogs, which are fitted by a normalizing flow model.
    \frac{dN}{dd_L dm_{1d} dm_{2d} dt_d} = R_0 p_{NF} (m_{1s}, m_{2s}, z)\frac{dV_c}{dz} \frac{1} {1+z} \frac{1}   {|J_{d\rightarrow s}|}

    Parameters
    ----------
    cosmology_wrapper: class
       Wrapper for the cosmological model
    catalog_wrapper: class
       Wrapper for the source-frame masses and merger redshift distribution, 
       coming from the normalizing flows fit of BBH synthesis catalogs
    scale free: bool
       if true it computes the scale-free likelihood (it neglects R0)
    """
    def __init__(self, cosmology_wrapper, catalog_wrapper, scale_free):
        self.cw = cosmology_wrapper
        self.cat_w = catalog_wrapper
        self.scale_free = scale_free

        if scale_free:
            if self.cat_w.population_parameters == None:
               self.population_parameters = (self.cw.population_parameters)
            else:
             self.population_parameters = (self.cw.population_parameters + self.cat_w.population_parameters)
        else:
            if self.cat_w.population_parameters == None:
               self.population_parameters = (self.cw.population_parameters) + ['R0']
            else:
             self.population_parameters = (self.cw.population_parameters + self.cat_w.population_parameters) + ['R0']

        event_parameters = ["mass_1", "mass_2", "luminosity_distance"]

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()

    def update(self, **kwargs):
        """
        This method updates the population parameters coming from the wrappers
        """
        if self.cat_w.population_parameters == None:
          self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        else:
          self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
          self.cat_w.update(**{key: kwargs[key] for key in self.cat_w.population_parameters})
         
        if not self.scale_free:
            self.R0 = kwargs['R0']            

    def log_rate_PE(self, prior, **kwargs):
        """
        This method computes the weights for the PE posterior samples.
        """
    
        xp = get_module_array(prior)
        ms1, ms2, z = detector2source(kwargs["mass_1"], kwargs["mass_2"], kwargs["luminosity_distance"], self.cw.cosmology)
        log_dVc_dz = xp.log(self.cw.cosmology.dVc_by_dzdOmega_at_z(z) * 4 * xp.pi)
        log_weights = (self.cat_w.log_pdf( xp.log(ms1), xp.log(ms2), xp.log(z)) + log_dVc_dz 
                      - xp.log(prior) - xp.log(detector2source_jacobian(z, self.cw.cosmology))
                      - xp.log1p(z) - xp.log(ms1) - xp.log(ms2) - xp.log(z)) 
        if self.scale_free:
            log_out = log_weights
        else:
            log_out = log_weights + xp.log(self.R0)
        return log_out

    def log_rate_injections(self, prior, **kwargs):
        """
        This method computes the weights for the injection samples.
        """
        return self.log_rate_PE(prior, **kwargs)



class CBC_rate_NF_synthesis(object):
    r"""
    Wrapper for managing a mixture model of two BBH population synthesis catalogs using Normalizing Flows.

    This rate model parametrizes the BBH rate per year at the detector in terms of source-frame
    masses and redshift. 
    
    The differential rate is modeled as a mixture of the two populations:
    $$ \frac{dN}{dd_L dm_{1d} dm_{2d} dt_d} = R_{\rm tot} \left[ (1-\phi) p_{\rm NF,1}(m_{1s}, m_{2s}, z) + \phi p_{\rm NF,2}(m_{1s}, m_{2s}, z) \right] \frac{1}{|J_{d\rightarrow s}|} $$
    where $R_{\rm tot} = \text{rate\_int1} + \text{rate\_int2}$ is the total integrated merger rate in unit of events per year, defined below.

    Parameters
    ----------
    cosmology_wrapper: class
        Wrapper for the cosmological model, used to compute source-frame transformations and Jacobians.
    catalog_wrapper1: class
        Trained `one_catalog_wrapper` on the first BBH population (e.g., isolated binaries).
    catalog_wrapper2: class
        Trained `one_catalog_wrapper` on the second BBH population (e.g., dynamical clusters).
    rate_int1: float
        Integrated expected number of mergers per year for the first catalog over the observable universe.
        Computed as the integral of (dVc/dz)*[R_1(z)/(1+z)]dz.
    rate_int2: float
        Integrated expected number of mergers per year for the second catalog over the observable universe.
        Computed as the integral of (dVc/dz)*[R_2(z)/(1+z)]dz.
    phi: float
        Mixing fraction in [0,1]. Represents the fractional contribution of the second population to the 
        total rate. Using `phi = rate_int2/(rate_int1+rate_int2)` recovers the fiducial astrophysical mixture.
    scale_free: bool set on False
        This model accounts for the astrophysical rates provided by the population synthesis models (via `rate_int`). 
    """   

    def __init__(self, cosmology_wrapper, 
                 catalog_wrapper1, catalog_wrapper2,
                 rate_int1, rate_int2):
        
        self.cw = cosmology_wrapper
        self.cat_w1 = catalog_wrapper1
        self.rate_int1 = rate_int1
        self.cat_w2 = catalog_wrapper2
        self.rate_int2 = rate_int2
        self.rate_int_tot = self.rate_int1 + self.rate_int2
        self.scale_free = False
        self.population_parameters = self.cw.population_parameters + ['phi']             

        event_parameters = ["mass_1", "mass_2", "luminosity_distance"]

        self.PEs_parameters = event_parameters.copy()
        self.injections_parameters = event_parameters.copy()

    def update(self, **kwargs):
        """
        This method updates the population parameters.
        """
        self.cw.update(**{key: kwargs[key] for key in self.cw.population_parameters})
        self.phi = kwargs['phi']
        
    def log_rate_PE(self, prior, **kwargs):
        """
        This method computes the weights for the PE posterior samples.
        """    
        xp = get_module_array(prior)
        ms1, ms2, z = detector2source(kwargs["mass_1"], kwargs["mass_2"], kwargs["luminosity_distance"], self.cw.cosmology)
        ms1 = xp.clip(ms1, 1e-6, None)
        ms2 = xp.clip(ms2, 1e-6, None)
        z = xp.clip(z, 1e-6, None)

        log_weights_cat1 = (self.cat_w1.log_pdf(xp.log(ms1), xp.log(ms2), xp.log(z)) 
                      - xp.log(prior) - xp.log(detector2source_jacobian(z, self.cw.cosmology)) 
                      - xp.log(ms1) - xp.log(ms2) - xp.log(z))
                      
        log_weights_cat2 = (self.cat_w2.log_pdf(xp.log(ms1), xp.log(ms2), xp.log(z)) 
                      - xp.log(prior) - xp.log(detector2source_jacobian(z, self.cw.cosmology)) 
                      - xp.log(ms1) - xp.log(ms2) - xp.log(z)) 

        phi = xp.clip(self.phi, 1e-6, 1.0 - 1e-6)
        rate_int_tot = xp.clip(self.rate_int_tot, 1e-6, None)
   
        log_mix1 = xp.log(1 - phi) + xp.log(rate_int_tot)
        log_mix2 = xp.log(phi) + xp.log(rate_int_tot)

        log_weights = xp.logaddexp(log_weights_cat1 + log_mix1, log_weights_cat2 + log_mix2)

        return log_weights

    def log_rate_injections(self, prior, **kwargs):
        """
        This method computes the weights for the injection samples.
        """        
        return self.log_rate_PE(prior, **kwargs)
