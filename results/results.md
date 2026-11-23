## Results (6740 test examples)

| Model                       |   BLEU-4 |   ROUGE-L |   METEOR |   Distinct-1 |   Distinct-2 |   Sentiment coherence |   Emotion macro-F1 (7) |   Emotion acc (7) |   Emotion macro-F1 (3) |   Emotion acc (3) |
|:----------------------------|---------:|----------:|---------:|-------------:|-------------:|----------------------:|-----------------------:|------------------:|-----------------------:|------------------:|
| Baseline                    |     2.73 |     15.89 |    15.39 |         4.08 |        18.66 |                 53.12 |                  18.6  |             54.69 |                  38.66 |             56.34 |
| EmoToken                    |     3.05 |     16.57 |    16.2  |         4.09 |        19.11 |                 51.74 |                  27.95 |             63.18 |                  47.54 |             64.38 |
| EmoToken (no <s> in prompt) |     2.77 |     15.85 |    15.37 |         4.1  |        18.89 |                 53.01 |                  19.23 |             52.92 |                  38.9  |             54.64 |
| EmoFusion                   |     1.85 |     15.24 |    14.21 |         4.5  |        19.42 |                 50.74 |                  28.44 |             65.16 |                  50.17 |             66.69 |
| Human reference             |   nan    |    nan    |   nan    |       nan    |       nan    |                nan    |                  28.38 |             63.74 |                  49.02 |             65.91 |
| Always neutral              |   nan    |    nan    |   nan    |       nan    |       nan    |                nan    |                  12.78 |             80.92 |                  29.82 |             80.92 |

## Paired bootstrap, 7-class macro-F1

| Comparison                              |   Difference (pts) |   95% CI low |   95% CI high |   Share of resamples better |
|:----------------------------------------|-------------------:|-------------:|--------------:|----------------------------:|
| EmoToken vs Baseline                    |              9.33  |        7.72  |        11.045 |                       1     |
| EmoFusion vs Baseline                   |              9.818 |        7.883 |        11.597 |                       1     |
| EmoFusion vs EmoToken                   |              0.488 |       -1.279 |         2.224 |                       0.703 |
| EmoToken vs EmoToken (no <s> in prompt) |              8.693 |        7.417 |        10.187 |                       1     |
