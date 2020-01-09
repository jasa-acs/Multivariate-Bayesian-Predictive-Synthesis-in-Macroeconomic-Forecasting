# Multivariate Bayesian Predictive Synthesis in Macroeconomic Forecasting

# Author Contributions Checklist Form

## Data

### Abstract

Data consists of 6 macroeconomic time series data from the FRED database. The 6 series are monthly inflation, wage, unemployment, consumption, investment, and interest rate from 1986/1-2015/12.

### Availability

All data are publicly available from the FRED database.

### Description

Inflation: https://fred.stlouisfed.org/series/CPILFENS 
Wage: https://fred.stlouisfed.org/series/CEU0500000008 
Unemployment: https://fred.stlouisfed.org/series/UNRATE 
Consumption: https://fred.stlouisfed.org/series/PCE 
Investment: https://fred.stlouisfed.org/series/NAPMNOI 
Interest Rate: https://fred.stlouisfed.org/series/FEDFUNDS


## Code

### Abstract

The zip file contains the matlab file and function that runs multivariate BPS, with the agent forecasts included as well for the 1-step ahead forecasts examined in the paper.

### Description

The function inputs the agent forecasts and prior specification and outputs the forecast coefficients and variance, as well as the posterior smoothed coefficients and agent densities. The code follows the synthesis function specification of the paper.

BPSsim.m is a file that loads the agent forecast densities and calls the mBPS.m function (where mBPS(y,a_j,A_j,n_j,delta,m_0,C_0,n_0,s_0,burn_in,mcmc_iter) produces the posterior parameters needed to forecast and analyze) sequentially, outputting the predictive distributions and then computing the performance measures.

## Instructions for Use

### Reproducibility 

Running BPSsim.m computes the predictive performances for the 1-step ahead forecasts. The BPS outputs are computed within the mBPS function (i.e. running mBPS(y,a_j,A_j,n_j,delta,m_0,C_0,n_0,s_0,burn_in,mcmc_iter) produces the posterior parameters needed to forecast and analyze). MSFE and LPDR are then computed outside the function to compare with the results of the paper (Table 1 and Figures 2-3). On-line coefficients and retrospective posteriors for the coefficients and the agents are also saved (which is used to produce Figures 4-8).

Using Matlab on Intel Core i7-7700k CPU @4.20 GHz, each iteration (t) takes on average approximately 5 minutes, taking approximately 30 hours for the whole process to complete. Each iteration (t) can be parallelized for faster computation.
