# amused
AMUSED – Adam Mickiewicz University's tools for Sentiment analysis and Emotion Detection

## License
MIT

## Gold standard

You can help creating gold standard corpus by annotating batches of selected sentences with provied script `annotate.py`:
```
./annotate.py corpora/wl-test.100b<BATCH_NO>.txt corpora/wl-test.100b<BATCH_NO>a<ANNOTATION_NO>.tsv
```
e.g.:
```
./annotate.py corpora/wl-test.100b1.txt corpora/wl-test.100b1a1.tsv
```
