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
- **Conference:** ICLR 2024  
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

For *CPM* method you must first generate logs with desired config

```bash
python code/log_certify.py --base_classifier models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar --N 10000 --sigma 1.0
```

it will produce logs in a "indir", from there you can certify with the method of your choice

```bash
python code/log_certify.py --indir indir --mode Rmono --certif pearson_clopper --alpha 0.001
```

For *LVMRS* to run certification on CIFAR-10, use the following command:
```bash
python code/certify_lvmrs.py \
--sigma 1.00 --skip 1 --N0 100 --N 100000 --batch_size 200 \
--outfile [file to store certification results]
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


## Pretrained Checkpoints

Pretrained models for randomized smoothing can be downloaded from the repository of Cohen et al. (2019):

**Repo:** [https://github.com/locuslab/smoothing](https://github.com/locuslab/smoothing)

These checkpoints have been widely used for robustness certification under Gaussian noise and serve as a reference for evaluating new certification methods.

## Codebase and References

This repository is based on existing open-source implementations, primarily:

- **Cohen et al. (2019):** _"Certified Adversarial Robustness via Randomized Smoothing"_  
  📄 **Paper:** [https://arxiv.org/abs/1902.02918](https://arxiv.org/abs/1902.02918)  
  💻 **Code:** [https://github.com/locuslab/smoothing](https://github.com/locuslab/smoothing)

- **(Certified!!) Adversarial Robustness for Free! (2022)**  
  📄 **Paper:** [https://arxiv.org/abs/2206.10550](https://arxiv.org/abs/2206.10550)  
  💻 **Code:** [https://github.com/ethz-spylab/diffusion_denoised_smoothing](https://github.com/ethz-spylab/diffusion_denoised_smoothing) 
