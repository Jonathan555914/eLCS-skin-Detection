# Skin Cancer eLCS Project

This project trains a local source-code copy of `scikit-eLCS` on handcrafted
skin-image features.

## Project Structure

```text
project/
├── code/
│   ├── scikit-eLCS/        # local GitHub source for eLCS
│   ├── extract_features.py # image -> feature CSV
│   └── train_lcs.py        # outlier removal, normalization, eLCS training
├── handcrafted_features/   # LBP, HOG, DWT, RGB/HSV, GLCM extractors
├── metadata/
│   ├── data/images/        # benign/ and malignant/ image folders
│   ├── features/           # extracted feature CSV files
│   └── results/            # experiment results
└── docs/                   # demo notes, weekly plan, research notes
```

## Workflow

Expected image layout:

```text
metadata/data/images/
├── benign/
└── malignant/
```

Extract features:

```bash
python code/extract_features.py metadata/data/images --output metadata/features/ALL/skin_features.csv --features all
```

Extract the two additional feature families separately:

```bash
python code/extract_features.py metadata/data/images --output metadata/features/RGB/rgb_features.csv --features rgb
python code/extract_features.py metadata/data/images --output metadata/features/GLCM/glcm_features.csv --features glcm
```

Combine them:

```bash
python code/extract_features.py metadata/data/images --output metadata/features/ALL/rgb_glcm_features.csv --features rgb+glcm
```

Train local eLCS:

```bash
python code/train_lcs.py metadata/features/ALL/skin_features.csv --class-column Class
```

## Important Notes

- `code/scikit-eLCS` is the main eLCS algorithm source.
- Do not edit a global pip-installed eLCS package.
- Outlier rows are removed before normalization.
- The scaler is fit on training data only, then applied to test data.
- Use `docs/DEBUG_DEMO_GUIDE.md` for the instructor demo walkthrough.
