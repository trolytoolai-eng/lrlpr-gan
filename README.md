# LRLPR: sharpen with a GAN, then read the license plate

Course project for Computer Vision (IT4343E), Hanoi University of Science and Technology.
Student: Đào Anh Quân (20261211M). Supervisor: M.Sc. Nguyễn Duy Tùng.
Tiếng Việt: [README_vi.md](README_vi.md)

**Problem.** Read a 7-character license plate from 5 tiny, blurry photos of the same car (dataset LRLPR-26, ICPR 2026 competition).

**Method.** Sharpen each image with a GAN (LPSRGAN), read it with an OCR (GP-LPR), then vote over the 5 readings.
The main idea is to let the frozen OCR grade the sharpening network during training (a real OCR loss with gradients).

## Results (test set, 3,000 cars, Exact Match = all 7 characters correct, voting over 5 images)

| Branch | What it is | Exact Match |
|---|---|---|
| M0 | Enlarge the LR image | 0.6% |
| M1 | Classical image processing | 0.6% |
| M2 | Published LPSRGAN, no extra training | 4.0% |
| M3 | Fine-tune (L1 + VGG19) | 4.9% |
| M4 | Fine-tune + data augmentation | 5.5% |
| M5c | Add the OCR loss (w = 3.0) | 53.1% |
| M5c+ | M5c, with the OCR retrained on GAN images | 58.7% |
| M6a | Network that looks at all 5 images at once | 55.5% |
| **M6a+** | M6a, with the retrained OCR | **60.6%** (95% CI 58.9% to 62.4%) |

Full tables, confidence intervals, McNemar tests, analysis and limitations are in the report.
Every number in the report is recomputed from the per-car prediction files, not typed by hand.

## What is in this repository

```
config/
  default.yaml            paths, seed, number of data-loading workers
  LRLPR/                  one yaml per branch: m3, m4, m5a, m5b, m5c, m6a, ocr_ft, ocr_sr
lrlpr/
  datasets/               read tracks, preprocessing (3:1 padding, CLAHE), augmentation, Dataset classes
  models/                 load GP-LPR and LPSRGAN, widen the first layer for multi-frame, patch the upstream repo
  losses/                 VGG19 perceptual loss
  train_funcs/            helpers for training (validation, cosine learning rate, OCR cache)
  utils/                  constants, common helpers, yaml reader
  utils_test/             evaluation, 5-frame fusion, metrics, report figures
split.py                  split the data into train / val / test
train_ocr.py              fine-tune the GP-LPR OCR on HR images
train.py                  fine-tune the sharpening network (M3, M4, M5, M6); settings come from config/LRLPR/*.yaml
train_ocr_sr.py           retrain the OCR on GAN outputs (branches with a + sign)
evaluate.py               evaluate one branch on val or test
show_track.py             show one example (images of all branches side by side and what the OCR read)
run_all.py                run the whole pipeline in three rounds
notebooks/
  LRLPR_colab.ipynb       short notebook that runs the tidy code on Google Colab
  run_history/            the three notebooks that were really run (outputs kept), see below
```

## The notebooks

The three notebooks in `notebooks/run_history/` are the real Colab runs behind the numbers in the report (Tesla T4, 13 November 2026).
Their outputs (logs, tables, figures) are kept so the run history can be checked.
They use the code **before** it was reorganized (one flat module), and Drive folder `MyCV`.
The code in this repository is the same code in a tidier layout.

| Notebook | Round | What it does | Run time on T4 |
|---|---|---|---|
| `LRLPR_Round1_full_pipeline.ipynb` | 1 | Split, baselines M0 to M2, OCR fine-tune, M3, M4, choose on val, test once | about 43 min |
| `LRLPR_Round2_ocr_loss_ocr_retrain.ipynb` | 2 | OCR loss (M5a, M5b, M5c), OCR retraining (the + branches), test once | about 102 min |
| `LRLPR_Round3_multiframe.ipynb` | 3 | Network that looks at 5 images (M6a), test once | about 27 min |

## The dataset is NOT in this repository

LRLPR-26 is released by its authors for academic, non-commercial research only. Its license says that redistribution,
in whole or in part, without the authors' permission is not allowed, and that the images may only be shown in academic publications and presentations.
So this repository contains **no dataset files** (no zip, no images, no labels, no per-car predictions with answers).

To get the data, follow the instructions on the official page:
<https://github.com/raysonlaroca/lrlpr-26-dataset> (sign the license agreement and send it from a university e-mail account).

After you get `LRLPR-26.zip`, you can check that it is the same file that was used for the report:

```
size   : 738,164,264 bytes
sha256 : 4c2177880223d280beb6c6fd114d2008a47976e2fd6eb7adb0a62657182c5395
content: train/Scenario-A/{Brazilian,Mercosur}/track_*  (5 LR + 5 HR images + annotations.json)
         train/Scenario-B/{Brazilian,Mercosur}/track_*
         test/track_*                                   (5 LR images + annotations.json)
         20,000 training tracks and 3,000 test tracks
```

You also need the GP-LPR OCR weights `Rodosol.pth` (released by the GP-LPR authors; not included here).

## How to run on Google Colab (Tesla T4)

1. Put `LRLPR-26.zip` and `models/Rodosol.pth` into `MyDrive/IT4343E/` (the "My Drive" section, not "Computers", because Colab cannot read that one).
2. Open `notebooks/LRLPR_colab.ipynb`, choose a T4 GPU and run the cells. Or in one cell:

```bash
git clone https://github.com/trolytoolai-eng/lrlpr-gan /content/lrlpr-gan
cd /content/lrlpr-gan
pip install -r requirements.txt
python run_all.py --round all
```

Colab may disconnect, so it is safer to run one round at a time (`--round 1`, then `--round 2`, then `--round 3`).
Results are saved on Drive, so a later round reuses the earlier ones.

### Useful options

```bash
python run_all.py --round 1 --stage m3,eval_m3        # only some steps of round 1
python run_all.py --round 2 --only m5c                # only train branch M5c
python run_all.py --round 1 --set epochs=1 --set workers=0   # very fast test run
python run_all.py --round 3 --set m6a.init_from=M5a   # override one branch only
```

| Round | Steps (`--stage`) |
|---|---|
| 1 | split, baselines, ocr, m3, eval_m3, m4, eval_m4, choose, test, report, bundle |
| 2 | check, m5, eval_m5, ocr_sr, eval_plus, choose, test, report, bundle |
| 3 | check, m6, eval_m6, choose, test, report, bundle |

Show one example (for questions):

```bash
python show_track.py --results_dir /content/drive/MyDrive/IT4343E/results --track test/track_10081 --split test --best M6a --out example.png
```

## Evaluation rules

- The model and the fusion method are chosen on the val set only (`choose`). Then the test set is run.
- The test set is run once per round. A lock file (`evidence/test_run.lock`, then `_v2`, `_v3`) records the time and the choice. A second run is refused.
- `evidence/` also keeps the time of each step, environment information, and sha256 hashes of the files.

## Differences from the upstream LPSRGAN code

When reading the LPSRGAN code (commit `68fc78a`, <https://github.com/valfride/lpsrgan>) we found these points and handled them here:

- The upstream OCR loss only counts wrongly read plates, so it gives no gradient. Here we use the cross-entropy of the frozen OCR, and the gradient flows back into the sharpening network (`train.py`, function `step_fn`).
- The adversarial term of the upstream code is commented out, so it really trains with L1 and VGG only. We keep this behavior (`adv_weight: 0`); it can be turned on again.
- The upstream augmentation shrinks the HR image when the rotation is chosen. We wrote our own function so that the HR image is always the true answer (`lrlpr/datasets/preprocess.py`, `augment_pair`).
- The upstream model loader swallows errors silently. Here loading is strict and checked after loading (`lrlpr/models/loaders.py`).
- The upstream repository is downloaded at the exact commit and two import lines (TensorFlow, basicsr) are patched at run time, see `lrlpr/models/setup_repo.py`. The authors' code is not copied into this repository.

## Notes on the results

- This code was reorganized from the code that ran the three Colab rounds. Function bodies are unchanged (checked automatically with syntax trees). We only added yaml reading and merged the three notebooks into one place.
- In round 1, the validation subset used while training took the first 500 tracks (all Brazilian daytime images). From round 2 the code samples randomly from all four groups (`stratified_subset`). The current code uses the second way for all three rounds, so rerunning round 1 may differ slightly from the report.
- We have not rerun everything on Colab after the reorganization. We ran all three rounds on small fake data to check the flow.
- Known limits: the OCR used in the loss and the OCR used for scoring are the same (GP-LPR); the M0+ baseline (OCR retrained on unsharpened LR images) is missing; each setting was run with one seed.

## Main references

- Pan, Tang, Tjahjadi, LPSRGAN, Neurocomputing 580, 2024.
- Nascimento et al., LCOFL, SIBGRAPI 2024, arXiv:2408.15103.
- Laroca et al., ICPR 2026 Competition on Low-Resolution License Plate Recognition, arXiv:2604.22506.
- Liu et al., GP-LPR, MMM 2024.
