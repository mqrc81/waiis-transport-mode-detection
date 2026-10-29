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

## ExtraSensory Dataset

- Used in: `experiments/02_family-baseline_shl-es`, `experiments/04_frozen-benchmark_es-shl` (benchmark `waiis_bench_v1`).
- Location: `<data root>/raw/extrasensory/` (zips and their unzipped folders). Source: http://extrasensory.ucsd.edu/
  (dataset page saved as `The ExtraSensory Dataset.html` in the same folder).
- Terms: the dataset page states no licence. It requires that published work using the dataset cites the original
  publication: Y. Vaizman, K. Ellis, G. Lanckriet, "Recognizing Detailed Human Context In-the-Wild from Smartphones
  and Smartwatches", IEEE Pervasive Computing 16(4), 2017. With no licence stated, treat it as research use only and
  ask the authors (UCSD) before any commercial use or redistribution.
- Files (sha256 checked 2026-10-28):

  | file (URL under http://extrasensory.ucsd.edu/data/) | bytes | sha256 |
  |---|---|---|
  | `primary_data_files/ExtraSensory.per_uuid_features_labels.zip` | 225,374,973 | `412d307bdf54a60dff81a705c2f25833889a46dca64a184e3b588ad99c69cb56` |
  | `additional_data_files/ExtraSensory.per_uuid_original_labels.zip` | 992,570 | `7c017f75cbf3ca6d157a6ecfd8aacb1173493d9864772ab648ba27ac81373e79` |
  | `additional_data_files/ExtraSensory.per_uuid_absolute_location.zip` | 2,303,359 | `09d06c2bf1f69d306b2e529d50d5fe2c8c5d56a832ddd3b64c8ee3e4908b750e` |
  | `raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip` | 6,475,399,720 | `61a487d027db28a3dc6ae5ea96eeda01b158dbba37b9d3e6439508b5cf323c01` |
  | `raw_measurements/ExtraSensory.raw_measurements.proc_gyro.zip` | 9,333,898,324 | `4bcff45bf008414956c5242a605b848813be38984a8f43f5dbcc461c13506e67` |
  | `raw_measurements/ExtraSensory.raw_measurements.raw_magnet.zip` | 4,065,885,563 | `3c1d8eed42eb28ccbd17c8277ec693676356d3bb9df4d1bd8608fb95a63932c9` |
  | `cv5Folds.zip` | 10,803 | `66183635452d8c1b250b570fba3000296bf29e4d54eda6af270d2b4f5a3b9ee6` |
