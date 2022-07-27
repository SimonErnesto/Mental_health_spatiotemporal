# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
import arviz as az
import geopandas as gpd
import matplotlib.pyplot as plt

##plotting parameters
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.titlesize': 14})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

path = "/lkj_model/"

os.chdir(path)

geod = gpd.read_file('south.geojson')

data = pd.read_csv("mental_health_covid_data.csv")

data = data[['country__text__text', 'Date', 'GPS_Tot', 'Subregion']]

data.columns = ['country', 'date', 'score', 'region']

data['country'].replace(' ', np.nan, inplace=True)
data['region'].replace(' ', np.nan, inplace=True)
data = data.dropna()

data.reset_index(inplace=True)

date = []
nums = np.arange(10).astype('str')
for i in range(len(data.date)):
    d = data.date[i]
    if d[0] != '0' and d[1] not in nums:
        d2 = '0'+d[0]+d[1:]
        date.append(d2)
    else:
        date.append(d)

date2 = []
nums = np.arange(10).astype('str')
for i in range(len(date)):
    d = date[i]
    if d[-7] == '/':
        d2 = d[:-6]+'0'+d[-6:]
        date2.append(d2)
    else:
        date2.append(d)

data['date'] = date2
    
data = data[data.region=='213']
data.reset_index(inplace=True)
data = data.groupby(['date', 'country'], as_index=False).sum()
# data = data.groupby(['date'], as_index=False).agg({'country':'first', 'score':'mean'})

data = data.sort_values('date')


D = len(data.date.unique())
C = len(data.country.unique())

c = pd.factorize(data['country'])[0].astype('int32')
d = pd.factorize(data['date'])[0].astype('int32')

ts = pd.unique(d)

score = data.score.values
zscores = (score-score.mean())/score.std()

# ######### GRW LKJ Moedl ##############
with pm.Model() as mod:
    sd = pm.HalfNormal.dist(1.0)
    L, corr, std = pm.LKJCholeskyCov("L", n=C, eta=2.0, sd_dist=sd, compute_corr=True) 
    Σ = pm.Deterministic("Σ", L.dot(L.T))  
    w = pm.Normal('w', 0, 1.0, shape=(D,C))  
    σ = pm.HalfNormal('σ', 1.0)  
    t_sq = tt.sqrt(ts)  
    β = pm.Deterministic('β', w.T*σ*t_sq)   
    B = pm.Deterministic('B', pm.math.matrix_dot(Σ,β))  
    α = pm.Normal('α', 0, 1.0, shape=C)  
    μ = pm.Deterministic('μ', α[c] + B[c,d]) 
    ϵ = pm.HalfNormal('ϵ', 1.0)
    y = pm.Normal("y", mu=μ, sigma=ϵ, observed=zscores)  

with mod:
    ppc = pm.sample_prior_predictive()

date_labels = [data.date[0], data.date[44], data.date[89], data.date[133], data.date[177]]
xticks = [0, 44, 89, 133, 177]

priord = ppc['y']
np.random.shuffle(priord)
for p in priord[450:]:
    plt.plot(p, alpha=0.1, color='g')
plt.plot(p, alpha=0.2, color='g', label='Prior Predictions')
plt.plot(zscores, color='purple', label='Observed')
plt.title('Prior Predictives')
plt.xticks(ticks=xticks, labels=date_labels)
plt.tick_params(labelsize=11)
plt.ylabel('GPS z-score')
plt.xlabel('Questionnaire Date')
plt.grid()
plt.legend()
plt.savefig('prior_predictions.png', dpi=300)
plt.close()

# with mod:
#     trace = pm.sample(2000, tune=2000, chains=4, cores=12, init='adapt_diag', target_accept=0.99)

tracedir = path+"trace/"
# pm.backends.ndarray.save_trace(trace, directory=tracedir, overwrite=True)

with mod:
    trace = pm.load_trace(tracedir)
    
# summ = az.summary(trace)
# summ.to_csv('summary.csv')

post = trace['μ']
pmean = post.mean(axis=0)
h5,h95 = np.array([az.hdi(post[:,i], hdi_prob=0.9) for i in range(len(post.T))]).T
labels_pos = [data.date[0], data.date[99], data.date[177]]
ticks_pos = [0, 62, 124]
plt.plot(d,zscores, color='k', alpha=0.5, label='Observed')
plt.plot(d,pmean, color='r', label='Posterior Mean')
plt.fill_between(d,h5,h95, color='r', alpha=0.3, label='90% HDI')
plt.title('Posterior Estimate')
plt.xticks(ticks=ticks_pos, labels=labels_pos)
plt.tick_params(labelsize=11)
plt.ylabel('GPS z-score')
plt.xlabel('Questionnaire Date')
plt.grid()
plt.legend()
plt.savefig('mu_posterior.png', dpi=300)
plt.close()

with mod:
    preds = pm.sample_posterior_predictive(trace)

pred = preds['y']
pred_mean = pred.mean(axis=0)
psd = pred.std(axis=0)
upper = pmean+psd
lower = pmean-psd

pred = preds['y']
np.random.shuffle(pred)
for p in pred[3950:]:
    plt.plot(p, alpha=0.1, color='orange')
plt.plot(p, alpha=0.3, color='orange', label='Posterior Predictions')
plt.plot(zscores, color='k', label='Observed')
plt.title('Posterior Predictions')
plt.xticks(ticks=xticks, labels=date_labels)
plt.tick_params(labelsize=11)
plt.ylabel('GPS z-score')
plt.xlabel('Questionnaire Date')
plt.grid()
plt.legend()
plt.savefig('posterior_predictions.png', dpi=300)
plt.close()

corrs = trace['L_corr'].mean(axis=0)

chile_corrs = trace['L_corr'].mean(axis=0)[2]

data['mu_mean'] = pmean
data['pred_mean'] = pred_mean
data['zscore'] = zscores

countries = data.groupby('country', as_index=False).mean()

countries = countries.sort_values('country')
geod = geod.sort_values('sovereignt')

geod2 = pd.concat([geod, countries.reindex(geod.index)], axis=1)
geod2['chile_corrs'] = chile_corrs

fig, ax = plt.subplots(2, 2, figsize=(10,10))
geod2.plot(column="zscore", ax=ax[0,0], legend=True)
geod2.plot(column="mu_mean", ax=ax[0,1], legend=True)
geod2.plot(column="pred_mean", ax=ax[1,0], legend=True)
geod2.plot(column="chile_corrs", ax=ax[1,1], legend=True, cmap='Wistia')
ax[0,0].set_title( 'Observed')
ax[0,1].set_title( 'Posterior mean')
ax[1,0].set_title( 'Predicted mean')
ax[1,1].set_title( 'Correlations')
plt.savefig('map_dists.png',dpi=300)



data_date = data.groupby('date', as_index=False).mean()

###plot rank
path_tranks = path+"tranks/"
varias = [v for v in trace.varnames if not "__" in v]
for var in tqdm(varias):
    err = az.plot_rank(trace, var_names=[var], kind='vlines', ref_line=True,
                       vlines_kwargs={'lw':1}, marker_vlines_kwargs={'lw':2})
    plt.savefig(path_tranks+var+'_trank.png', dpi=300)
    plt.close()

