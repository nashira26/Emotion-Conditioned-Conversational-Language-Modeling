# Emotion-Conditioned-Conversational-Language-Modeling
Standard dialogue systems can produce responses
that make sense in context, but they often fail to express
emotions appropriately. This paper explores whether adding
explicit emotion information during fine-tuning of a transformerbased sequence-to-sequence model can improve the emotional
quality of responses. We fine-tune BART using the full DailyDialog
dataset with emotion labels at the utterance level. The model is
evaluated against a baseline model without emotion conditioning,
using metrics such as semantic quality, emotion alignment, and
sentiment consistency. Results show that the emotion-conditioned
model performs better, achieving a 3-class emotion alignment
accuracy of 54.70% and a macro-F1 score of 40.25% compared
to 50.95% and 37.12% for the baseline. This suggests that adding
emotion information improves emotional appropriateness without
harming the overall meaning of responses. We also compare two
methods for incorporating emotion: EmoToken, which adds a
learned emotion token at the decoder input, and EmoFusion, which
integrates emotion embeddings into the encoder. Our findings show
that encoder-side fusion (EmoFusion) provides a more effective
and stable way to include emotion, especially when the training
data is imbalanced
