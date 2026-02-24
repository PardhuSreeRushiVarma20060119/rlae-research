<img width="1920" height="1080" alt="RLAE SVAR" src="https://github.com/user-attachments/assets/d581cd67-3c90-493f-b5e6-e739729aaed1" />

# On the Structural Limitations of Weight-Based Neural Adaptation and the Role of Reversible Behavioral Learning

This repository contains the implementation and experimental framework
accompanying the paper:

**On the Structural Limitations of Weight-Based Neural Adaptation and
the Role of Reversible Behavioral Learning.**

- **Zenodo DOI:** [10.5281/zenodo.18761938](https://doi.org/10.5281/zenodo.18761938)
  
> [!NOTE]
> Cite all versions? You can cite all versions by using the DOI : [10.5281/zenodo.18738128](https://doi.org/10.5281/zenodo.18738128).  
> This DOI represents all versions and will always resolve to the latest one.

- **arXiv Link:** Coming Soon!
- **TMLR Journal Publication:** Coming Soon! 

It is provided strictly for reproducibility. 

------------------------------------------------------------------------

## Scope

This repository implements:

-   Runtime Low-Rank Adaptive Environments (RLAE)
-   Structural Variance Analysis for Robustness (SVAR)

The base model remains permanently frozen.\
All adaptive behavior is implemented via LoRA modules.\
SVAR provides diagnostic evaluation only.

------------------------------------------------------------------------

## Model Configuration

-   Base model: Qwen2.5-3B-Instruct
-   Quantized execution supported
-   No base weight mutation

------------------------------------------------------------------------

## Reproducibility

Install dependencies:

    pip install -r experiments/requirements.txt

Run baseline and adaptation pipeline:

    python src/exp1_reset/1_baseline.py
    python src/exp1_reset/2_train_sft.py
    python src/exp1_reset/3_train_rl.py

Verify reset integrity:

    python src/exp1_reset/4_verify_reset.py

Run structural diagnostics:

    python src/exp3_svar/perturbation.py
    python src/verification/robustness_suite.py

------------------------------------------------------------------------

## Repository Structure

    arts/
    colab-experiments/
    experiments/
    Papers/

------------------------------------------------------------------------

## Contribution Policy

This repository is a research artifact accompanying the published
paper. External contributions are not accepted.

------------------------------------------------------------------------

## License

GNU Affero General Public License v3.0 (AGPL-3.0)

------------------------------------------------------------------------

> *“Intelligence as powerful and alive, yet deliberately hollow at its center — governed, observable, and stripped of its identity.”*
