# HJM_calibration_toy_project

## This project to develop and calibrated multi-factor model using HJM framework to estimate the expected exposure at default and CVA of interest rate derivatives and other fixed-income products. 
Overview
- HJM.py: Contains HJM_Model class (debugging and model improvement are in progress).
- Gaussian HJM.ipynb: Shows results of fitting simple models (multi-factor models are in progress).
- utils.py: Contains general functions (e.g., yield curve conversion and implied volatility calculation with root finding method).
- test_cases.py: Contains simple unit test cases. (I use PyTest for unit testing)

Reference:
- Angelini, F. and Herzel, S., 2005. Consistent calibration of HJM models to cap implied volatilities. Journal of Futures Markets: Futures, Options, and Other Derivative Products, 25(11), pp.1093-1120.
- Björk, T. and Christensen, B.J., 1999. Interest rate dynamics and consistent forward rate curves. Mathematical Finance, 9(4), pp.323-348.