# BERT transformer

## BERT
**BERT -- Bidirectional Encoder Representations from Transformers**

### Word Embeddings
*Basic idea -- the words with similar meaning are situated closely in the vector space*

Problem: word embeddings does not encount context of the sentence
Solution: train contextual representation on text.
> Example:
> * bank in "open a bank accound": [..., 0.9, -0.2, 1.6, ...]
> * bank in "on the river bank": [..., -1.9, -0.4, 0.1, ...]