# Unified Metrics Summary (Golden 200 / Synthetic 4-Class)

## 3-Class Metrics (vs golden_dataset_200)

*(Excluding rows where golden 3-class label is "none"; only alt / both / base.)*

```
                             Method      Variant  accuracy  f1_macro   n   f1_alt  f1_both  f1_base
                    XGBoost 3-class          raw     0.440  0.352189 200 0.475728 0.459627 0.121212
XGBoost per-attr ensemble (3-class)          raw     0.505  0.277435 200 0.000000 0.676056 0.156250
         Phase5 confidence baseline          raw     0.620  0.391008 200 0.293333 0.784452 0.095238
             Phase5 XGBoost 3-class          raw     0.585  0.265881 200 0.000000 0.743590 0.054054
                    SLM (Gemma3 4B)          raw     0.155  0.089466 200 0.000000 0.000000 0.268398
                         SLM (Kimi)          raw     0.235  0.222808 200 0.343750 0.000000 0.324675
                  SLM (GPT-4o mini)          raw     0.155  0.089466 200 0.000000 0.000000 0.268398
                   SLM Recalculated recalculated     0.570  0.399120 200 0.225352 0.730627 0.241379
                         Google API recalculated     0.260  0.147927 200 0.411523 0.032258 0.000000
                    Scrape + search recalculated     0.270  0.227767 200 0.070175 0.333333 0.279793
```

- **Best F1 (macro) 3-class:** SLM Recalculated (recalculated) = 0.3991
- **Best Accuracy 3-class:** Phase5 confidence baseline (raw) = 0.6200

## 4-Class Metrics (vs synthetic_4class_golden)

```
                             Method      Variant  accuracy  f1_macro   n  f1_none   f1_alt  f1_base  f1_both
                    SLM (Gemma3 4B) recalculated  0.436170  0.200414 376 0.000000 0.000000 0.189655 0.612000
                         SLM (Kimi) recalculated  0.433511  0.239250 376 0.000000 0.160000 0.185185 0.611814
                  SLM (GPT-4o mini) recalculated  0.404255  0.247252 376 0.000000 0.282609 0.136986 0.569412
                    XGBoost 4-class          raw  0.380319  0.236168 376 0.000000 0.198347 0.188406 0.557920
XGBoost per-attr ensemble (4-class)          raw  0.406915  0.186839 376 0.027778 0.000000 0.126984 0.592593
                    SLM (Gemma3 4B)          raw  0.132979  0.069444 376 0.000000 0.000000 0.277778 0.000000
                         SLM (Kimi)          raw  0.151596  0.127147 376 0.000000 0.272300 0.236287 0.000000
                  SLM (GPT-4o mini)          raw  0.000000  0.000000 376 0.000000 0.000000 0.000000 0.000000
                         Google API recalculated  0.178191  0.087206 376 0.000000 0.297674 0.029412 0.021739
                    Scrape + search recalculated  0.220745  0.150626 376 0.000000 0.077922 0.272480 0.252101
```

- **Best F1 (macro) 4-class:** SLM (GPT-4o mini) (recalculated) = 0.2473
- **Best Accuracy 4-class:** SLM (Gemma3 4B) (recalculated) = 0.4362

## Binary Metrics (vs golden_dataset_200 2class_testlabels)

```
                            Method      Variant  accuracy       f1   n
                    XGBoost binary          raw     0.430 0.601399 200
              Random Forest binary          raw     0.560 0.102041 200
XGBoost per-attr ensemble (binary)          raw     0.565 0.043956 200
                   SLM (Gemma3 4B)          raw     0.570 0.000000 200
                        SLM (Kimi)          raw     0.655 0.576687 200
                 SLM (GPT-4o mini)          raw     0.570 0.000000 200
                  SLM Recalculated recalculated     0.450 0.607143 200
```

- **Best F1 (binary):** SLM Recalculated = 0.6071

## Similarity (value similarity vs golden)

```
         Method  mean_overall_similarity   n
     Google API                33.602399 200
Scrape + search                63.349911 200
```

### Per-Attribute Similarity

Mean value similarity (0–100) vs golden per attribute.

```
         Method  phone  web  address  category   n
     Google API   13.0  5.0    82.81       0.0 200
Scrape + search   92.5  0.5    97.05       0.0 200
```

## Per-Attribute Winner Agreement

```
         Method  agreement_pct   n
     Google API           21.5 200
Scrape + search           53.0 200
```

## Per-Attribute Accuracy

Accuracy per attribute (method winner vs golden `attr_*_winner`) for methods with per-attribute outputs.

```
                   Method phone_acc web_acc address_acc category_acc   n
               Google API      9.5%   16.5%       22.5%        37.5% 200
          Scrape + search     27.0%   59.5%       68.5%        57.0% 200
          SLM (Gemma3 4B)     20.0%   69.0%       40.0%        51.5% 200
               SLM (Kimi)     45.5%   57.5%       70.5%        71.5% 200
        SLM (GPT-4o mini)     34.5%   48.0%       68.5%        65.5% 200
XGBoost per-attr ensemble     11.5%   70.5%       45.5%        51.5% 200
```

## Readability / Informativeness

Mean number of filled attributes (phone, web, address, category) per row on golden:
- **Base (source):** 2.98
- **Alt (conflated):** 2.95
- **Which has more information:** Base

