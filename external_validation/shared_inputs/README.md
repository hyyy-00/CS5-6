# Processed external price input

`inputs/test40_with_prices.csv` supplies the prices and weekly rows used by the frozen Ensemble strategy replay. The default runner joins it to regenerated predictions by date and ticker.

This directory contains no standalone model fitting or simulation. The Ensemble training demonstration reads only the processed training data under `data/reproducible/`.
