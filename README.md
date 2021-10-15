# amused
AMUSED – Adam Mickiewicz University's tools for Sentiment analysis and Emotion Detection

## License
MIT

## Gold standard

You can help creating gold standard corpus by annotating batches of selected sentences with provied script `annotate.py`:
```
./annotate.py corpora/wl-test.100b<BATCH_NO>.txt
```
e.g.:
```
./annotate.py corpora/wl-test.100b1.txt
```

## Preparation for installation and running experiments

I recommend using [venv](https://docs.python.org/3/library/venv.html):
```
python3 -m venv venv
source ./venv/bin/activate
```

Install requirements:
```
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Install [Morfeusz](http://morfeusz.sgjp.pl/download)

Install [PyTorch](https://pytorch.org/get-started/locally)

Install [Transformers](https://huggingface.co/transformers/installation.html)

## Installation

```
python ./setup.py install
```

## Running experiments

```
mkdir experiments_results
python ./experiments.py
```