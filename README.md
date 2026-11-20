# Accompanying Python Notebooks for *From Hard to Soft Alignments: A Tutorial on Soft Dynamic Time Warping for Music Processing*

© [Meinard Müller](https://www.audiolabs-erlangen.de/fau/professor/mueller) and [Johannes Zeitler](https://audiolabs-erlangen.de/fau/assistant/zeitler) 2026.

These notebooks provide a hands-on introduction to Soft Dynamic Time Warping (SDTW) for music processing. They combine compact explanations, visualizations, and exercises with small Python examples, moving from differentiable minimum functions to training a multipitch estimation model from weakly aligned data. The material is intended for students familiar with signal processing and basic Python/PyTorch. No prior knowledge of SDTW is required. All examples run on a standard CPU.

Start with the [course overview](00_overview.ipynb). The five units build on one another:

1. [Differentiable Approximations of Minimum Functions](01_min_functions.ipynb): Explore function values, gradients, and the role of the temperature parameter.
2. [Soft Alignments and the SDTW Gradient](02_soft_alignment.ipynb): Interpret soft alignments and compare the global formulation with dynamic programming.
3. [Training with Strong Targets](03_training_strong.ipynb): Introduce the dataset, CNN and sigmoid, frame-wise BCE, and precision, recall, and F-measure.
4. [Training with Weak Targets: Basic](04_training_weak_basic.ipynb): Train SDTW with unfolded weak targets; study batching, modified targets, and temperature scheduling.
5. [Training with Weak Targets: Advanced](05_training_weak_advanced.ipynb): Train SDTW with compact targets; investigate step sizes and weights, loss offsets, and temporal shifts.



If you use these notebooks for your work, please cite the accompanying article:
```
@article{MuellerZ27_eduSDTW_TISMIR,
    author = {Meinard M{\"u}ller and Johannes Zeitler},
    title  = {From Hard to Soft Alignments: A Tutorial on Soft Dynamic Time Warping for Music Processing},
    year   = {2027},
    journal = {Transactions of the International Society for Music Information Retrieval},
    note   = {Under Review}
    }
```

## Installation

Create and activate a Conda environment, then install the dDTW package:

```bash
conda create -n edu_sdtw python=3.12 pip
conda activate edu_sdtw
python -m pip install "ddtw[benchmark,notebook]" scipy
```

The [benchmark,notebook] options install all needed additional dependencies. See the [*d*DTW docu](https://groupmm.github.io/dDTW/installation.html#nvidia-gpu-linux) for details about a cuda-compatible installation.

## Further Software & Code
The following section provides useful links to alignment-related software packages and accompanying code for experiments on SDTW from different publications. For users interested in applying SDTW and its variants, we primarily refer to the [*d*DTW Toolbox](https://github.com/groupmm/ddtw).
### Software
* *d*DTW: A Unified and Efficient Toolbox for Differentiable Sequence Alignment [[Github]](https://github.com/groupmm/ddtw)
* PyTorch CTC loss [[API docu]](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CTCLoss.html)
* Sync Toolbox: Reference implementations for offline DTW [[Github]](https://github.com/groupmm/synctoolbox)

### Related code for experiments

* J. Zeitler and M. Müller: A Unified Perspective on CTC and SDTW using Differentiable DTW [[GitHub]](https://github.com/groupmm/dDTW_CTC)
* J. Zeitler and M. Müller: Subsequence Soft Dynamic Time Warping [[Github]](https://github.com/groupmm/subsequenceSDTW)
* J. Zeitler, M. Krause, and M. Müller: Soft Dynamic Time Warping With Variable Step Weights [[GitHub]](https://github.com/groupmm/weightedSDTW)
* J. Zeitler, S. Deniffel, M. Krause, and M. Müller: Stabilizing Training With Soft Dynamic Time Warping [[GitHub]](https://github.com/groupmm/stabilizing_sdtw)
* J. Zeitler and M. Müller: Reformulating Soft Dynamic Time Warping: Insights Into Target Artifacts and Prediction Quality [[Website]](/resources/MIR/2025_ZeitlerM_softDTW_modTargets_ISMIR)



## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Grant No. 500643750 (MU 2686/15-1) and Grant No. 521420645 (MU 2686/17-1). The authors are with the International Audio Laboratories Erlangen, a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.


