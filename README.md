# Emotion-Conditioned Dialogue Generation with BART

Dialogue models usually produce replies that make sense in context but ignore emotion. This project fine-tunes BART on DailyDialog and compares two ways of telling the model which emotion its reply should have:

- **EmoToken:** an emotion token such as `[EMOTION-HAPPINESS]` starts the decoder output.
- **EmoFusion:** a learned emotion vector is added to every encoder input embedding.

Both are compared with a **baseline** that gets no emotion information. All three are trained from the same pretrained weights, on the same data, and evaluated on the same test set.

For the architecture, design choices and limitations, see [DESIGN.md](DESIGN.md).

## Results

*Add the tables from `outputs/results.md` here after running the notebook.*

## How to run

Everything is in one notebook, [`EmoCon.ipynb`](EmoCon.ipynb). It needs no files from you: it downloads DailyDialog, trains the three models, and evaluates them.

1. Open `EmoCon.ipynb` in Google Colab (upload it, or open it from GitHub).
2. Choose Runtime → Change runtime type → **T4 GPU**.
3. Choose Runtime → **Run all**, and allow access to Google Drive when asked.

Everything is saved to `MyDrive/EmoCon/`:

```
MyDrive/EmoCon/
├── config.json             settings used for the run
├── data/                   the downloaded dataset and the prepared examples
├── checkpoints/            baseline/, emotoken/, emofusion/
└── outputs/                generated responses, results.md, results.csv, bootstrap.csv
```

**If Colab disconnects:** open the notebook again and choose Run all. Finished models are skipped, an unfinished model resumes from its last completed epoch, and saved outputs are reused. Training three models takes several hours in total, so expect to do this at least once on the free tier. Each epoch's time is printed, so after the first epoch you can estimate the rest.

**If the dataset download fails:** download `ijcnlp_dailydialog.zip` from http://yanran.li/dailydialog, put it in `MyDrive/EmoCon/data/`, and run the notebook again. The notebook also tries a copy on the Hugging Face Hub before asking for this.

## Setup

| | |
|---|---|
| Model | `facebook/bart-base` (139M parameters), full fine-tuning |
| Data | DailyDialog, official train / validation / test split (11,118 / 1,000 / 1,000 dialogues) |
| Examples | Context of up to 3 previous turns; target emotion = the response turn's label |
| Training | 3 epochs, AdamW (lr 3e-5), effective batch 32, mixed precision, best epoch by validation loss |
| Class imbalance | Square-root inverse-frequency weights: on the emotion token (EmoToken) or per example (EmoFusion) |
| Generation | Beam search, 4 beams, no repeated 3-grams |
| Emotion classifier | `j-hartmann/emotion-english-distilroberta-base` |
| Main metric | Emotion macro-F1, with a paired bootstrap for significance |
| Hardware | Google Colab, T4 GPU |

## Repository

```
├── EmoCon.ipynb       the whole project: data, training, generation, evaluation
├── README.md
├── DESIGN.md          architecture, design choices, limitations, next steps
└── requirements.txt   for running outside Colab
```
