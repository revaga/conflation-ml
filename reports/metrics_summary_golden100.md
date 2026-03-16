# Unified Metrics Summary (Golden 100)

## 3-Class Metrics (vs golden_dataset_200)

*(Excluding rows where golden 3-class label is "none"; only alt / both / base.)*

```
                    Method      Variant  accuracy  f1_macro   n   f1_alt  f1_both  f1_base
           XGBoost 3-class          raw      0.56  0.445253 100 0.666667 0.509091 0.160000
Phase5 confidence baseline          raw      0.44  0.338317 100 0.285714 0.591304 0.137931
    Phase5 XGBoost 3-class          raw      0.34  0.193040 100 0.000000 0.507692 0.071429
         SLM (GPT-4o mini)          raw      0.23  0.124661 100 0.000000 0.000000 0.373984
                SLM (Kimi)          raw      0.38  0.307353 100 0.475000 0.000000 0.447059
     SLM (GPT-4o mini alt)          raw      0.23  0.124661 100 0.000000 0.000000 0.373984
         SLM (GPT-4o mini) recalculated      0.33  0.230841 100 0.000000 0.487395 0.205128
                SLM (Kimi) recalculated      0.41  0.353898 100 0.264151 0.527273 0.270270
     SLM (GPT-4o mini alt) recalculated      0.49  0.408889 100 0.506667 0.560000 0.160000
                Google API recalculated      0.41  0.196643 100 0.589928 0.000000 0.000000
           Scrape + search recalculated      0.32  0.292177 100 0.088889 0.415094 0.372549
```

- **Best F1 (macro) 3-class:** XGBoost 3-class (raw) = 0.4453
- **Best Accuracy 3-class:** XGBoost 3-class (raw) = 0.5600

## 4-Class Metrics (vs synthetic_4class_golden)

```
               Method      Variant  accuracy  f1_macro   n  f1_none   f1_alt  f1_base  f1_both
    SLM (GPT-4o mini) recalculated  0.436170  0.200414 376      0.0 0.000000 0.189655 0.612000
           SLM (Kimi) recalculated  0.433511  0.239250 376      0.0 0.160000 0.185185 0.611814
SLM (GPT-4o mini alt) recalculated  0.404255  0.247252 376      0.0 0.282609 0.136986 0.569412
      XGBoost 4-class          raw  0.457447  0.195733 376      0.0 0.000000 0.155556 0.627376
    SLM (GPT-4o mini)          raw  0.132979  0.069444 376      0.0 0.000000 0.277778 0.000000
           SLM (Kimi)          raw  0.151596  0.127147 376      0.0 0.272300 0.236287 0.000000
SLM (GPT-4o mini alt)          raw  0.000000  0.000000 376      0.0 0.000000 0.000000 0.000000
           Google API recalculated  0.178191  0.087206 376      0.0 0.297674 0.029412 0.021739
      Scrape + search recalculated  0.220745  0.150626 376      0.0 0.077922 0.272480 0.252101
```

- **Best F1 (macro) 4-class:** SLM (GPT-4o mini alt) (recalculated) = 0.2473
- **Best Accuracy 4-class:** XGBoost 4-class (raw) = 0.4574

## Binary Metrics (vs golden_dataset_200 2class_testlabels)

```
               Method      Variant  accuracy       f1   n
       XGBoost binary          raw      0.51 0.675497 100
 Random Forest binary          raw      0.49 0.105263 100
    SLM (GPT-4o mini) recalculated      0.53 0.651852 100
    SLM (GPT-4o mini)          raw      0.49 0.000000 100
           SLM (Kimi) recalculated      0.51 0.642336 100
           SLM (Kimi)          raw      0.61 0.561798 100
SLM (GPT-4o mini alt) recalculated      0.53 0.684564 100
SLM (GPT-4o mini alt)          raw      0.49 0.000000 100
```

- **Best F1 (binary):** SLM (GPT-4o mini alt) = 0.6846

## SLM (multiple) — Recalculated vs Raw

Accuracy comparison: labels **recalculated** from attribute winners vs **raw** record-level SLM output.

**Binary truth = 2class_testlabels** (from attr_*_winner tie-breaking):

| Metric | Recalculated | Raw (GPT-4o mini) | Raw (Kimi) | Raw (GPT-4o mini alt) |
|--------|--------------|-------------------|------------|------------------------|
| 3-Class Accuracy | 49.0% | 23.0% | 38.0% | 23.0% |
| Binary Accuracy | 53.0% | 49.0% | 61.0% | 49.0% |

**Binary truth = 3-class → binary** (alt+both→alt, base→base; matches reference table):

| Metric | Recalculated | Raw (GPT-4o mini) | Raw (Kimi) | Raw (GPT-4o mini alt) |
|--------|--------------|-------------------|------------|------------------------|
| Binary Accuracy | 79.0% | 23.0% | 53.0% | 23.0% |

## Similarity (value similarity vs golden)

```
         Method  mean_overall_similarity   n
     Google API                19.082611 100
Scrape + search                45.314562 100
```

## Per-Attribute Winner Agreement

```
         Method  agreement_pct   n
     Google API          26.75 100
Scrape + search          49.25 100
```

## Per-Attribute Accuracy

Accuracy per attribute (method winner vs golden `attr_*_winner`) for methods with per-attribute outputs.

```
               Method phone_acc web_acc address_acc category_acc   n
           Google API     18.0%   12.0%       38.0%        39.0% 100
      Scrape + search     28.0%   57.0%       54.0%        58.0% 100
    SLM (GPT-4o mini)     15.0%   69.0%       33.0%        46.0% 100
           SLM (Kimi)     53.0%   60.0%       58.0%        69.0% 100
SLM (GPT-4o mini alt)     42.0%   47.0%       54.0%        68.0% 100
```

## Readability / Informativeness

Mean number of filled attributes (phone, web, address, category) per row on golden:
- **Base (source):** 2.96
- **Alt (conflated):** 2.92
- **Which has more information:** Base

