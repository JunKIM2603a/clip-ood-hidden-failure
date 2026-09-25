# H2 prompt-sensitivity analysis

Primary H2 uses the 8 meaning-preserving templates already frozen in configs/pilot.yaml.

Failure is not redefined from the prompt ensemble. It is the original H1 baseline
false positive under the original method-specific ImageNet ID95 threshold.

Primary comparison:

- M0: error ~ z(mu)
- M1: error ~ z(mu) + z(sigma)
- association support: bootstrap 95% CI for beta_sigma is strictly above 0
- predictive support: 5-fold OOF log-loss improvement M0-M1 has bootstrap 95% CI strictly above 0
- condition support requires both criteria

| Method | Source | Errors | beta_sigma | OR / 1SD sigma | delta OOF log-loss | delta OOF AUROC | Supported |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| mcm | inaturalist | 3396 | -0.5312 | 0.5879 | 0.008028 | 0.001605 | no |
| mcm | sun | 4056 | -0.2221 | 0.8009 | 0.000847 | 0.000194 | no |
| neglabel | inaturalist | 339 | 0.0008 | 1.0008 | -0.000034 | -0.000022 | no |
| neglabel | sun | 2271 | -0.0029 | 0.9971 | -0.000129 | -0.000105 | no |

**H2 decision: FAIL**

- eligible conditions: 4
- supported conditions: 0
- required supported conditions: 3
- project gate after H2: CONDITIONAL_GO

H3 is intentionally not evaluated by this stage.
