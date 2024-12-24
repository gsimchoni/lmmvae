# LMMVAE - Integrating Random Effects in VAE

This is the code repository for our paper "Integrating Random Effects in Variational Autoencoders for Dimensionality Reduction of Correlated Data": https://arxiv.org/pdf/2412.16899

Simulations may be run with the `simulate.py` script using different configuration files like so:

```bash
python simulate.py --conf conf_files/conf_categorical.yaml --out res.csv
```

Real data experiments are included in the `notebooks` folder, R scripts to reproduce the visualizations in the paper are included in the `r_scripts` folder.