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

## NOR-TMD (candidate for waiis_bench_v2; experiment 08)

- Used in: `experiments/08_candidate-datasets-qc` (quality check).
- Source: Kaggle `scholarone/nor-tmd` (https://www.kaggle.com/datasets/scholarone/nor-tmd), version 1, updated
  2025-01-12; one file `nor_tmd.csv` (73,611,929,573 bytes; zip archive 7,359,659,643 bytes).
  Location: `<data root>/raw/nor_tmd/nor-tmd.zip` (kept zipped; downloaded 2026-11-02).
- Licence: the Kaggle page states **CC BY 4.0** (Attribution 4.0 International). The article is CC BY-NC 4.0; that
  licence covers the article, not the data. The publisher page (tandfonline.com) returned 403, so its data
  availability statement was not read; the paper's text (thesis copy) says "This dataset has been made publicly
  available on Kaggle (Skretting & Grønli, 2025)".
- **Commercial use: verified as permitted on 2026-11-02.** Attribution (below) is required under CC BY 4.0.
- Credit (requested on the Kaggle page): A. Skretting, T.-M. Grønli, R. R. Mukkamala, "Feature importance from
  novel dataset for on-device transport mode detection", Journal of Intelligent Transportation Systems 30(4), 2025,
  doi:10.1080/15472450.2025.2515567. Dataset reference: A. Skretting, T.-M. Grønli (2025), NOR-TMD dataset for
  transport mode detection, Kaggle.

## MobilitApp Sensor Dataset (candidate for waiis_bench_v2; experiment 08)

- Used in: `experiments/08_candidate-datasets-qc` (inventory only; verdict "not usable", approved 2026-11-02).
- Source: CORA.RDR (dataverse.csuc.cat), doi:10.34810/data1948, version 1.0 (released 2024-12-24). Location:
  `<data root>/raw/mobilitapp/` (`MobilitApp-SensorDataset-OK.zip`, `Readme.txt`).
- Licence: **CC0 1.0** (Dataverse metadata and `Readme.txt`). No attribution required; cite anyway.
- Credit: M. Aguilar Igartua, G. Caravaca Ibáñez, M. Gotanegra Estañol, "MobilitApp - Mobility sensor dataset",
  CORA.Repositori de Dades de Recerca, 2024, doi:10.34810/data1948. Related: G. Caravaca Ibáñez, "Exploring the
  capabilities of deep learning models for transport and human activity recognition", Master's thesis, UPC, 2024,
  http://hdl.handle.net/2117/400780.

## Collecty dataset (candidate for waiis_bench_v2; experiment 08)

- Used in: `experiments/08_candidate-datasets-qc` (quality check).
- Source: Kaggle `merdelic/dataset-for-multimodal-transport-analytics`, version 4, updated 2023-07-15,
  doi:10.34740/kaggle/dsv/4976711 (as given in the article); 454 trajectory files plus a consent form
  (24,864,805,329 bytes unzipped; zip archive 5,904,835,122 bytes). Location:
  `<data root>/raw/collecty/collecty.zip` (kept zipped; downloaded 2026-11-02).
- Licence: the Kaggle page states **"Unknown"**. The data article is CC BY 4.0 (it covers the article; the article
  says the data "can be downloaded upon registering").
- **Commercial use: verified as permitted on 2026-11-02** (the Kaggle page still shows "Unknown"). Cite the article.
- Credit: M. Erdelić, T. Erdelić, T. Carić, "Dataset for multimodal transport analytics of smartphone users -
  Collecty", Data in Brief 50, 109481, 2023, doi:10.1016/j.dib.2023.109481.
