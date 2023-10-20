from pymc_experimental.model_builder import ModelBuilder
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from pathlib import Path
import arviz as az
import pymc as pm
import pymc_bart as pmb
import xarray as xr
import json
import scipy.stats as sp
import warnings

from numpy.random import RandomState
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

class BartSurvModel(ModelBuilder):
    # Give the model a name
    _model_type = "BART_Survival"

    # And a version
    version = "0.1"

    def build_model(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, **kwargs):
        # check x,y,weights        
        # X_values = X
        # y_values = y
        # add in the data processing code
        self._generate_and_preprocess_model_data(X, y, weights)
        # if type(split_rules is not None):
        #     self.model_config["split_rules"] = split_rules
        # Get model Configs
        SPLIT_RULES = [eval(rule) for rule in self.model_config.get("split_rules", None)]
        M = self.model_config.get("trees", 20)

        # custom logp
        def logp_bern(value, mu, w):
            return w * pm.logp(pm.Bernoulli.dist(mu), value)
        # extension of the bernoulli, the distribution is used as normal
        def dist_bern(mu, w, size):
            return pm.Bernoulli.dist(mu, size=size)
        off = sp.norm.ppf(np.mean(self.y))
        # M = 20 # number of trees

        
        with pm.Model() as self.model:    
            x_data = pm.MutableData("x_data", self.X)
            w = pm.MutableData("weights", self.weights)
            # change names of y_values
            f = pmb.BART("f", X=x_data, Y=self.y.flatten(), m=M, split_rules = SPLIT_RULES)
            z = pm.Deterministic("z", (f + off))
            mu = pm.Deterministic("mu", pm.math.invprobit(z))
            pm.CustomDist("y_pred", mu, w.flatten(), dist=dist_bern, logp=logp_bern, observed=self.y.flatten(), shape = x_data.shape[0])            
    
    def sample_model(self, **kwargs):
        if self.model is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first or call .fit() instead."
            )
        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}
            idata = pm.sample(**sampler_args)
            # idata.extend(pm.sample_prior_predictive())
            idata.extend(pm.sample_posterior_predictive(idata, var_names=["mu"]))

        idata = self.set_idata_attrs(idata)
        return idata


    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray=None,
        progressbar: bool = True,
        predictor_names: List[str] = None,
        random_seed: RandomState = None,
        **kwargs: Any,
    ) -> az.InferenceData:

        self.build_model(X, y, weights)
        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)
        self.idata = self.sample_model(**sampler_config)
        self.is_fitted_ = True
        self.predictor_names = predictor_names

        # X_df = pd.DataFrame(X, columns=X.columns)
        # combined_data = pd.concat([X_df, y], axis=1)
        combined_data = np.hstack([self.y, self.weights, self.X])
        # assert all(combined_data.columns), "All columns must have non-empty names"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning#,
                # message="The group fit_data is not defined in the InferenceData scheme",
            )
            # self.idata.add_groups(fit_data=combined_data.to_xarray())  # type: ignore
            self.idata.add_groups(fit_data=xr.DataArray(combined_data))  # type: ignore
            self.idata.add_groups(predictor_names = xr.DataArray(self.predictor_names))
        return self.idata  # type: ignore


    def sample_posterior_predictive(self, X_pred, extend_idata, combined, **kwargs):
        """Predict new data"""
        # self._data_setter(X_pred)
        print(self.model["x_data"].eval().shape)

        with self.model:  # sample with new input data
            pm.set_data({"x_data":X_pred})
            print(self.model["x_data"].eval().shape)

        with self.model:
            post_pred = pm.sample_posterior_predictive(self.idata, var_names=["mu"], model = self.model,**kwargs)
            if extend_idata:
                self.idata.extend(post_pred)

        posterior_predictive_samples = az.extract(
            post_pred, "posterior_predictive", combined=combined
        )

        # return posterior_predictive_samples
        return post_pred
    
    def _data_setter(
        self, X:np.ndarray, y:np.ndarray = None
    ):
        with self.model:
            pm.set_data({"x_data": X})
            if y is not None:
                pm.set_data({"y_data": y.values if isinstance(y, pd.Series) else y})


    @classmethod
    def load(cls, fname: str):
        filepath = Path(str(fname))
        idata = az.from_netcdf(filepath)
        # needs to be converted, because json.loads was changing tuple to list
        model_config = cls._model_config_formatting(json.loads(idata.attrs["model_config"]))
        model = cls(
            model_config=model_config,
            sampler_config=json.loads(idata.attrs["sampler_config"]),
        )
        model.idata = idata
        dataset = idata.fit_data["x"].values
        X = dataset[:,2:]
        y = dataset[:,0]
        weights = dataset[:,1]
        model.build_model(X, y, weights)

        # All previously used data is in idata.
        cls.predictor_names = idata.predictor_names["x"].values
        if model.id != idata.attrs["id"]:
            raise ValueError(
                f"The file '{fname}' does not contain an inference data of the same model or configuration as '{cls._model_type}'"
            )

        return model


    @staticmethod
    def get_default_model_config() -> Dict:
        print("NO DEFAULT MODEL CONFIGS, MUST SPECIFY")
        pass

    @staticmethod
    def get_default_sampler_config() -> Dict:
        sampler_config: Dict = {
            "draws": 100,
            "tune": 100,
            "cores": 2,
            "chains": 2,
            "compute_convergence_checks": False
        }
        return sampler_config

    @property
    def output_var(self):
        return "y_pred"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        model_config = self.model_config.copy()
        model_config["split_rules"] = [str(sp_rule) for sp_rule in model_config["split_rules"]]
        return model_config

    def _save_input_params(self, idata) -> None:
        # idata.attrs["weights"] = json.dumps(self.weights.tolist())
        pass

    def _generate_and_preprocess_model_data(
        self, X: np.ndarray, 
        y:np.ndarray,
        weights:np.ndarray
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        self.model_coords = None  # in our case we're not using coords, but if we were, we would define them here, or later on in the function, if extracting them from the data.
        # as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
        # and usually we would need to do some preprocessing, or generate the coords from the data.
        self.X = X
        self.y = y.reshape(y.shape[0],1)
        if weights is None:
            weights = np.ones(X.shape[0])
        self.weights = weights.reshape(weights.shape[0],1)