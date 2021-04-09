# swiss-bert

Code for training Swiss German BERT models using the SwissCrawl corpus.

## Requirements

- The [simpletransformers](https://simpletransformers.ai/) library
- The [SwissCrawl](https://icosys.ch/swisscrawl) corpus
- The [bert-base-german-cased](https://huggingface.co/dbmdz/bert-base-german-cased) and/or [bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased) models provided by DBMDZ
- We used the VarDial 2020 SMG training data (without dialect labels) as a validation set, but any other validation data can be used instead.

## Train BERT from scratch on SwissCrawl

Extract the plain text from SwissCrawl:
`tail -n+2 swisscrawl-2019-11-23.csv | cut -f 1 > swisscrawl_reformatted.txt`

Specify whether to train a cased or uncased model, and specify the vocabulary size:
`python3 train_from_scratch.py cased|uncased 30000`

We tested uncased models with vocabulary sizes of 3000 and 30000. The latter yielded better results.

# Continue pretraining a DBMDZ BERT with SwissCrawl

This is the recommended setup and yields better results than training from scratch.

Extract the plain text from SwissCrawl:
`tail -n+2 swisscrawl-2019-11-23.csv | cut -f 1 > swisscrawl_reformatted.txt`

Load existing model and continue training on SwissCrawl:
`python3 train_from_existing.py cased|uncased`

