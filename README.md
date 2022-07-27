<h1> Gaussian Random Walk and Gaussian Process applied to Global Mental Health data </h1>

<p>
	The present repository contains results from spatiotemporal models applied to results from a mental health questionnaire used to asses mental health globally during the covid-19 pandemic (Olff et al., 2022). Models are simply a proof of concept, so they were applied to the south-American region only.
</p>
<p></p>

<h1> Model 1 </h1>

<p> The first model implements a multivariate Gaussian random walk (GRW) prior with an LKJ prior for covariances and correlations. </p>

<p align="center"> &#987; ~ HalfNormal(1) </p>
<p align="center"> L, R, SD ~ LKJ(n=E, &eta;=6, sd=&#987;)</p>
<p align="center"> &Sigma; = LL<sup>T</sup> </p>
<p align="center"> w<sub>d,c</sub> ~ Normal(0,1) </p>
<p align="center"> &sigma; ~ HalfNormal(1) </p>
<p align="center"> t<sub>d</sub> = &Sqrt;time<sub>d</sub>... &Sqrt;time<sub>D</sub> </p>
<p align="center"> &beta;<sub>d,c</sub> = w<sub>d,c</sub>t<sub>d</sub>&sigma; </p>
<p align="center"> B = &Sigma;&beta;<sub>d,c</sub> </p>
<p align="center"> &alpha;<sub>c</sub> ~ Normal(0, 1) </p>
<p align="center"> &mu;<sub>c</sub> = &alpha;<sub>c</sub> + B </p>
<p align="center"> &varepsilon;<sub>c</sub> ~ HalfNormal(0.05) + 1 </p>
<p align="center"> y<sub>d,c</sub> ~ Normal(&mu;<sub>d,c</sub>, &varepsilon;) </p>

<p> Where c…C , C = 7, is the number of countries, d…D, D = 178, is the number of dates when questionnaires were taken. Countries are only countries which provided data within the south-American region. Questionnaires were administered between April and November 2020 (i.e. dates). Note that the GRW is expressed as the product of a standard Gaussian w, a standard deviation &sigma;, and the square roots of times (see Morokof, 1998). </p>

<p>  The model was sampled using Markov chain Monte Carlo (MCMC) No U-turn sampling (NUTS) with 2000 tuning steps, 2000 samples, 4 chains. </p>

<h1> Results </h1>

<p> Prior predictive checks are a bit wild, but models with narrower predictions sample worse. </p>

<p align="center">
	<img src="lkj_model/prior_predictions.png" width="800" height="500" />
</p>

<p> Posterior estimates show a good approximation and uncertainty. </p>

<p align="center">
	<img src="lkj_model/mu_posterior.png" width="800" height="500" />
</p>


<p> Predictions from the posterior also indicate sensible and reasonable uncertainty. </p>

<p align="center">
	<img src="lkj_model/posterior_predictions.png" width="800" height="500" />
</p>

<h1> Conclusion </h1>

<p> Although the model performs well in terms of inference and predictions, showing a reasonable/sensible measurement of uncertainty, the sampling is not ideal, with some parameters showing rather low ESS (above 200 but below 1000). </p>


<h1> Model 2  </h1>
<p> The second model implements a multivariate Gaussian process (GP) prior with an exponential quadratic covariance function. The model is intended for comparison, so it is rather simple. </p>

<p align="center"> &ell; ~ HalfNormal(1) </p>
<p align="center"> k(x, x') = exp[-(x,x')<sup>2</sup>/2&ell;<sup>2</sup>] </p>
<p align="center"> f(x) ~ GP(m(x), k(x, x'))
<p align="center"> &mu; = log(f(x)) </p>
<p align="center"> y ~ Poisson(&mu;) </p>

<p> Where k(x, x') is an exponential quadratic kernel covariance function, and m(x) is the mean function, x are the dates (sample points), and observed data (y) are the questionnaire aggregated scores. </p>

<p>  The model was sampled using Markov chain Monte Carlo (MCMC) No U-turn sampling (NUTS) with 1000 tuning steps, 1000 samples, 4 chains, with ADVI initialization. </p>

<h1> Results </h1>

<p> Prior predictive checks are relatively reasonable. </p>

<p align="center">
	<img src="gp_model/prior_predictions.png" width="800" height="500" />
</p>

<p> Posterior estimates show a good approximation and uncertainty (both posterior and data were re-scaled to z-scores). </p>

<p align="center">
	<img src="gp_model/f_posterior.png" width="800" height="500" />
</p>


<p> Predictions from the posterior seem to underestimate uncertainty in regions with low samples (i.e. dates/countries with fewer questionnaire data). </p>

<p align="center">
	<img src="gp_model/posterior_predictions.png" width="800" height="500" />
</p>

<h1> Conclusion </h1>

<p> This model sampled better in terms (all ESS over 1000), and has the advantage of not requiring data transformation; namely questionnaire scores (added scores by country, counts) do not need to be transformed. However, predictions cannot properly account for higher uncertainty in regions with lower density. </p>


<h1> References </h1>

Morokoff (1998). Generating Quasi-Random Paths for Stochastic Processes . https://www.jstor.org/stable/2653031

Olff et al (2022). Mental health responses to COVID-19 around the world. https://doi.org10.1080/20008198.2021.1929754
