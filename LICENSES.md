# Licences and terms of use

## SHL dataset (University of Sussex-Huawei Locomotion and Transportation Dataset)

- Used in: `experiments/01_unimts-fewshot_shl` (User 1, Hips, preview release v2).
- Location: `<data root>/raw/shl/`; full terms in `<data root>/raw/shl/LICENSE.txt`.
- Terms: publicly available **solely for non-profit and research purposes**. Any other use requires licensing terms
  from the originators (daniel.roggen@ieee.org).
- Credit: cite the publication currently recommended on shl-dataset.org. At download time that was:
  - H. Gjoreski, M. Ciliberto, L. Wang, F. J. Ordoñez Morales, S. Mekki, S. Valentin, D. Roggen, "The University of
    Sussex-Huawei Locomotion and Transportation Dataset for Multimodal Analytics with Mobile Devices", IEEE Access, 2018.
  - For the collection app: M. Ciliberto, F. J. Ordoñez Morales, H. Gjoreski, D. Roggen, S. Mekki, S. Valentin, "High
    reliability Android application for multidevice multimodal mobile data acquisition and annotation", ACM SenSys, 2017.
- Naming: use the full name "The University of Sussex-Huawei Locomotion and Transportation Dataset" at least once,
  then "the SHL Dataset".

## UniMTS (code and pretrained checkpoint)

- Used in: `experiments/01_unimts-fewshot_shl`.
- Source: https://github.com/xiyuanzh/UniMTS (clone in `../ext/UniMTS`), checkpoint `UniMTS.pth` from Hugging Face
  `xiyuanz/UniMTS` (`<data root>/checkpoints/unimts/checkpoint/`).
- Licence: Apache-2.0, for both code and checkpoint. Keep the licence and notices when redistributing; state changes
  (our changes: `experiments/01_unimts-fewshot_shl/patches/UniMTS.patch`).
  Source of the licence: the Hugging Face model card `xiyuanz/UniMTS` (`license: apache-2.0`; that repo also holds the
  code). The GitHub repo at commit 656fcda has no LICENSE file and no licence metadata (checked 2026-10-23).
- Credit: cite X. Zhang, D. Teng, R. R. Chowdhury, S. Li, D. Hong, R. K. Gupta, J. Shang, "UniMTS: Unified
  Pre-training for Motion Time Series", NeurIPS 2024 (arXiv:2410.19818).
