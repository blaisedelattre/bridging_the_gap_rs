# Repository for Lipschitz Randomized Smoothing

This repository contains the code for the following works:

- **Bridging the Theoretical Gap in Randomized Smoothing**  
- **The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing**

## Papers

### Bridging the Theoretical Gap in Randomized Smoothing

- **Title:** Bridging the Theoretical Gap in Randomized Smoothing  
- **Authors:** Blaise Delattre, Paul Caillon, Quentin Barthélemy, Erwan Fagnou, Alexandre Allauzen  
- **Conference:** AISTATS 2025  
- **Paper Link:** [OpenReview](https://openreview.net/forum?id=AZ6T7HdCRt)

### The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing

- **Title:** The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing  
- **Authors:** Blaise Delattre, Alexandre Araujo, Quentin Barthélemy, Alexandre Allauzen  
- **Conference:** ICLR, 2024  
- **Paper Link:** [OpenReview](https://openreview.net/forum?id=C36v8541Ns)

## Code

```
bridging_the_gap_rs/
├── LICENSE
├── counter_example.py              # Counter example Bonferroni   
├── code/
│   ├── improved_diffusion/         # Directory with DRM source
│   ├── DRM.py                      # Implementation of DRM (details in the paper LVMRS)
│   ├── architectures.py            # Network architectures (e.g. ResNet variants)
│   ├── core_cpm.py                 # Core routines for the CPM method
│   ├── datasets.py                 # Dataset handling and preprocessing
│   ├── train.py                    # Main training script
│   ├── train_utils.py              # Helper functions for training
│   ├── sparsemax.py                # Sparsemax activation implementation
│   ├── log_certify.py              # executable to produce certify log
│   ├── PUB/                        # Directory for PUB (Product Upper Bound) computations
│   ├── archs/                      # Directory containing network architecture definitions
│   ├── certify_lvmrs.py            # Certification script using LVMRS method
│   ├── certify_cpm_from_log.py     # Certification script from log files for CPM method
│   ├── core_lvmrs.py               # Core routines for LVMRS-based certification
│   ├── predict.py                  # Inference and prediction script
│   └── compute_rs_lipschitz_radius.py  # Computation of the Lipschitz constant for Lipschitz randomized smoothing
├── README.md                       # Repository overview and instructions
```

## Citation

If you find our work useful, please cite the papers as follows.

For *Bridging the Theoretical Gap in Randomized Smoothing*:
```bibtex
@inproceedings{
delattre2025bridging,
title={Bridging the Theoretical Gap in Randomized Smoothing},
author={Blaise Delattre and Paul Caillon and Erwan Fagnou and Quentin Barth{\'e}lemy and Alexandre Allauzen},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=AZ6T7HdCRt}
}
```

For *The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing*:

```bibtex
@inproceedings{
delattre2024the,
title={The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing},
author={Blaise Delattre and Alexandre Araujo and Quentin Barth{\'e}lemy and Alexandre Allauzen},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=C36v8541Ns}
}
```
