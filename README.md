# Capstone: Buying vs. Renting a Property in Toronto

Developing a financial tool targeted towards young adults to provide increased clarity on the economics behind buying or renting a property in today's housing market in Toronto.

Motivation:
- Home prices can be essentially impossible to predict due to the incredible number of factors that affects them. However, if we could find patterns in the individual factors that affect prices, we may be able to derive a prediction.

Methods: 
- XGBoost as Primary Predictor
- ARIMAX + Harmonic + Random Noise as Column Extender for 'backbone' variables (e.g. population, GDP)
- ARIMAX as Column Extender for Dependant variables
