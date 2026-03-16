# ELISA

**E**fficient **L**ikelihood **I**nference for **S**tellar **A**ges

A Python package for Bayesian inference of stellar cluster parameters from Gaia photometry.

[![Documentation](https://astroelisa.readthedocs.io/)](https://astroelisa.readthedocs.io/)

## What it does

ELISA provides two inference engines:

- **MCMC isochrone fitting** — infers cluster age, metallicity, distance modulus, and extinction by fitting PARSEC isochrones with `emcee`.
- **Binary-star SBI** — estimates primary mass and mass ratio for individual stars using Simulation-Based Inference (SNPE neural posterior estimation).

## Installation

```bash
git clone https://github.com/stevenalfonso/elisa.git
cd elisa
pip install -e .
```

For SBI support (requires PyTorch):

```bash
pip install torch sbi isochrones
```

## Documentation

Full usage guides and API reference are at **[astroelisa.readthedocs.io](https://astroelisa.readthedocs.io)**:

- [Installation](https://astroelisa.readthedocs.io/en/latest/installation.html)
- [Quick Start — MCMC](https://astroelisa.readthedocs.io/en/latest/quickstart.html)
- [Binary Star SBI](https://astroelisa.readthedocs.io/en/latest/sbi.html)
- [API Reference](https://astroelisa.readthedocs.io/en/latest/api.html)

## Contributing

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/stevenalfonso/elisa/issues).

To contribute code, fork the repository and open a pull request against `main`. Please:

- Include tests for new functionality (`pytest`)
- Follow the existing code style (`black`, `ruff`)
- Describe what the PR changes and why

## References

- PARSEC isochrones: [Bressan et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.427..127B)
- emcee: [Foreman-Mackey et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F)
- Likelihood formulation: [von Hippel et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006ApJ...645.1436V)

## License

MIT License — see [LICENSE](LICENSE) for details.
