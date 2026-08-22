# Validation Protocol

- Corpus: `experiments\data\wikitext2_raw\train.txt`
- Characters used: 500000
- Tokenizer: character-level, vocabulary size 127
- Validation batches: indices [301, 302, 303], 3 batches, 384 tokens total
- Sequence length: 128; batch size: 1
- Evaluation cadence: after every continuation optimizer update
- Candidate and clean continuations use identical validation batches.
- Validation-loss gap: `abs(validation_loss_clean - validation_loss_candidate)`.
