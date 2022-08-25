# Multilingual Captioning of Cardiac Signals

This is an implementation of the experiments conducted in the manuscript: "Let Your Heart Speak in its Mother Tongue: Multilingual Captioning of Cardiac Signals". 

This implementation includes:
1. Source code to pre-train a decoder network using RTLP, and various other methods (e.g., MARGE, MLM)
2. Source code to pre-train an encoder network using supervised learning
3. Source code to perform multilingual cardiac signal captioning (in 7 languages!)
4. Source code to evaluate and visualize the results (BLEU, ROUGE, etc.)

# Requirements

The code requires the following:

* Python 3.6 or higher
* PyTorch 1.0 or higher
* NumPy
* tqdm 
* SpaCy
* nltk
* Pandas
* Sklearn
* SciPy
* wfdb
* Seaborn

## Summary

We generate multilingual captions based on cardiac signals in the PTB-XL dataset. Since the PTB-XL dataset only contains reports in Enlgish, we first translate the reports into seven different languages to allow for a multilingual experimental setup. In this implementation, you can conduct the fine-tuning experiments based on an encoder which has been pre-trained in a supervised manner and a decoder pre-trained using RTLP.

## Datasets

### Download

The PTB-XL dataset can be downloaded from the following link:

1) [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/)

## Training

To train the model(s) in the paper, run this command:

```
python run_experiments.py
```

## Evaluation

To evaluate the model(s) in the paper, run this command:

```
python run_experiments.py
```
