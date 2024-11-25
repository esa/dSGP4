$\partial\textrm{SGP4}$ Documentation
================================

**dsgp4** is a differentiable SGP4 program written leveraging the [PyTorch](https://pytorch.org/) machine learning framework: this enables features like automatic differentiation and batch propagation (across different TLEs) that were not previously available in the original implementation. Furthermore, it also offers a hybrid propagation scheme called ML-dSGP4 where dSGP4 and ML models can be combined to enhance SGP4 accuracy when higher-precision simulated (e.g. from a numerical integrator) or observed (e.g. from ephemerides) data is available. 

For more details on the model and results, check out our publication: [Acciarini, Giacomo, Atılım Güneş Baydin, and Dario Izzo. "*Closing the Gap Between SGP4 and High-Precision Propagation via Differentiable Programming*" (2024) Vol. 226(1), pages: 694-701](https://doi.org/10.1016/j.actaastro.2024.10.063)


The authors are [Giacomo Acciarini](https://www.esa.int/gsp/ACT/team/giacomo_acciarini/), [Atılım Güneş Baydin](https://gbaydin.github.io/), [Dario Izzo](https://www.esa.int/gsp/ACT/team/dario_izzo/). The main developer is Giacomo Acciarini (giacomo.acciarini@gmail.com).

```{toctree}
:maxdepth: 1
:caption: Getting Started

install
capabilities
credits
```

```{toctree}
:maxdepth: 1
:caption: Contents

tutorials
api
```
