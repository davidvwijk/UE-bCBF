# Uncertainty Estimators for Robust Backup Control Barrier Functions

*Abstract*: 
Designing safe controllers is crucial and notoriously challenging for input-constrained safety-critical control systems. Backup control barrier functions offer an approach for the construction of safe controllers online by considering the flow of the system under a backup controller. However, in the presence of model uncertainties, the flow cannot be accurately computed, making this method insufficient for safety assurance. To tackle this shortcoming, we integrate backup control barrier functions with uncertainty estimators and calculate the flow under a reconstruction of the model uncertainty while refining this estimate over time. We prove that the controllers resulting from the proposed _Uncertainty Estimator Backup Control Barrier Function (UE-bCBF)_ approach guarantee safety, are robust to unknown disturbances, and satisfy input constraints.

David E. J. van Wijk, Ersin Das, Anil Alan, Samuel Coogan, Tamas G. Molnar, Joel W. Burdick, Manoranjan Majji, Kerianne L. Hobbs. "Uncertainty Estimators for Robust Backup Control Barrier Functions," Submitted to Automatica, 2025. Preprint: https://arxiv.org/abs/2503.15734

## Supplemental Video: Quadrotor Example
The objective is to prevent the quadrotor from crashing below the minimum altitude in the presence of unknown disturbances. Our approach obeys the altitude constraint by calculating the backup flow under a reconstruction of the disturbance. Further, since the backup controller is designed to satisfy input bounds, the safe controller is guaranteed to respect these bounds as well. Click the thumbnail below to watch!

[![Quadrotor Supplemental Video](https://github.com/davidvwijk/UE-bCBF/blob/main/thumbnail.png)](https://www.youtube.com/watch?v=btNq8rAtAkM&feature=youtu.be)

## BibTeX Citation

```
@article{vanwijk2025uncertaintyestimatorsrobustbackup,
    title={Uncertainty Estimators for Robust Backup Control Barrier Functions}, 
    author={David E. J. van Wijk and Ersin Da{\c{s}} and Anil Alan and Samuel Coogan and Tamas G. Molnar and Joel W. Burdick and Manoranjan Majji and Kerianne L. Hobbs},
    journal={arXiv preprint arXiv:2503.15734},
    year={2025},
}
```
