### pymc3
---
https://github.com/pymc-devs/pymc3

```py 
import pymc3 as pm

X, y = linear_training_data()
with pm.Model() as linear_model:
  weights = pm.Normal('weights', mu=0, sd=1)
  noise = pm.Gamma('noise', alpha=2, beta=1)
  y_observed = pm.Normal('y_observed',
    mu=X.dot(weights),
    sd=noise,
    observed=y)
  prior = pm.sample_prior_predictive()
  posterior = pm.sample()
  posterior_pred = pm.sample_posterior_predictive(posterior)
  
```

```
conda install -c conda-forge pymc3
pip install pymc3
pip install git+http://github.com/pymc-devs/pymc3

metaplotlib inline

pip install pymc3
conda install -c conda-forgepymc3
pip install pymc3
pip install git+https://github.com/pymc-devs/pymc3
pip install -r requirements.txt
```

```py
lp = Laplace.dist(mu=0, b=0.05)
x_eval = np.linspace(-.5, .5, 300)
plt.plot(x_eval, theano.tensor.exp(lp.logp(x_eval)).eval())
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Laplace distribution');

with Model() as model_lasso:
  priors = ['Intercept': Normal.dist(mu=0, sd=50),
    'Regressor': Laplace.dist(mu=0, b=0.05)
  ]
  GLM.from_formula('male ~height + weight', htwt_data, family=glm.families.Binomial(),
    priors=priors)
  trace_lasso = sample(500, cores=2)
  
trace_df = trace_to_dataframe(trace_lasso)
scatter_matrix(trace_df, figsize=(8, 8));
print(trace_df.describe().drop('count').T)
  
```

