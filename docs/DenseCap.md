## Requirements
- java 1.8+
- python 3 (Python2 has not been tested)
    - gensim
- Stanford CoreNLP 3.6.0[(download)](http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip)
    - add stanford-corenlp-3.6.0.jar to `pycocoevalcap/spice/lib/`
    - add stanford-corenlp-3.6.0-models.jar to `pycocoevalcap/spice/lib/`

- `google_word2vec_model` for WMD[(download)](https://docs.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM)
    - unzip it and add GoogleNews-vectors-negative300.bin to `pycocoevalcap/wmd/data`

## References
- WMD metric from [https://github.com/mtanti/coco-caption](https://github.com/mtanti/coco-caption)
- main code from [https://github.com/wangleihitcs/CaptionMetrics](https://github.com/wangleihitcs/CaptionMetrics)
