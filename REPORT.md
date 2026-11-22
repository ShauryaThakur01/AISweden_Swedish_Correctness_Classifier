# Does synthetic supervision transfer to Swedish grammaticality scoring?

## Approach

The task is to assign Swedish text a continuous grammaticality score between zero and one. This experiment asks whether examples produced by a small Swedish grammar can replace the supplied supervision when the encoder, training budget and validation set are held fixed. Both candidates use the cased KB-BERT encoder from the National Library of Sweden. Its Swedish pretraining and bidirectional context make it a plausible starting point for distinctions such as adjective agreement and verb placement. A single forward pass per batch also suits the assignment's throughput criterion. The starter system averages frozen encoder representations and fits a separate support vector regressor. Here, a one-output head and the encoder are trained together so that the representation can adapt to the scoring task. [KB-BERT model card](https://huggingface.co/KBLab/bert-base-swedish-cased)

## Data and supervision

The assignment provides 478 labeled training examples and 120 public test examples. The training records carry a continuous score, a good/poor label and a group identifier linking related examples. Holding out individual rows could place a corrected sentence in training and its corrupted counterpart in validation. The preparation procedure shuffles group identifiers with seed 42 and assigns approximately one fifth of the groups to validation. The supplied-data candidate uses the remaining rows and their original scores.

The synthetic candidate uses an equal number of newly generated texts, with the same counts of good and poor examples. Three frames combine people, times, objects and actions. For example, *I går läste läraren ett nytt brev* has the expected verb-second order and neuter agreement. The generator can produce *I går läraren läste ett nytt brev* by moving the subject ahead of the finite verb, or *I går läste läraren ett ny brev* by removing the adjective's agreement ending. Other frames allow an incorrect finite verb after a modal or infinitive marker. Clean examples receive 0.95, examples with one injected error receive 0.45, and examples with two receive 0.25. These values encode a severity assumption. No human rated the generated sentences. Case-folding and whitespace normalization filter exact text collisions with the supplied development data and within the generated set.

## Experimental design

Each run starts independently from the same immutable KB-BERT revision and seed. The tokenizer preserves case and truncates at 256 tokens. A sigmoid transforms the model's single logit into a bounded score. Training minimizes mean squared error against the continuous targets for five epochs using AdamW, a learning rate of 0.00002, weight decay of 0.01 and batches of eight. Matching training size and batch size gives each candidate the same number of parameter updates.

The primary validation metric is Spearman correlation because it measures whether the scorer orders examples by their reference scores. ROC-AUC measures separation of good and poor examples, and mean absolute error measures distance from the numerical targets. The final-epoch checkpoint with the higher validation Spearman advances to the public test. Ties are resolved by higher ROC-AUC, lower mean absolute error and then the supplied-data candidate. The starter evaluator times the entire prediction subprocess, including model loading and file I/O. Its samples-per-minute figure therefore describes a complete run at this test size.

## Results

| Training source | Validation Spearman | Validation ROC-AUC | Validation MAE |
|---|---:|---:|---:|
| Supplied | 0.832 | 0.978 | 0.105 |
| Generated | 0.562 | 0.797 | 0.237 |

Each candidate had 124,691,713 parameters and trained on 385 examples. On the 93 shared validation examples, the supplied-data model led by 0.270 in Spearman correlation and reduced mean absolute error from 0.237 to 0.105. It was selected for the public test.

On the 120 public test examples, the selected model achieved **0.832 Spearman and 0.953 ROC-AUC**. The evaluator recorded **12.3 seconds and 586 samples per minute**, including model loading. The run used Windows and PyTorch 2.6.0 on an AMD CPU (family 25, model 80). The assignment reports 0.528 Spearman, 0.783 ROC-AUC and 714 samples per minute for its starter baseline on this split. KB-BERT scored higher on both quality measures. The baseline used a separately reported laptop CPU, so the timings cannot isolate the effect of model choice.

## Limitations and error analysis

The generator covers three short sentence frames, all with neuter singular objects. A model could associate *ett ny* with poor scores yet struggle with agreement across a longer phrase. It also receives no explicit synthetic supervision for preposition choice, pronoun reference or discourse context. A fixed penalty for each injected error assumes that error counts track perceived severity. The single group split and seed leave uncertainty about how stable the ranking would be under resampling.

On the validation sentence *Hon köpte en ny vinter jacka.*, the reference score was 0.45 because *vinterjacka* should be one word. The supplied-data model scored it 0.473 and the synthetic model 0.879. Compound splitting is absent from the generator. Both models also underestimated the severity of combined errors in *Företaget har anställde tjugo nya medarbetarer i år.* The perfect construction requires *har anställt* and the plural form is *medarbetare*. Against a reference of 0.20, the models predicted 0.349 and 0.451, respectively. These examples illustrate two failure modes on the validation split. The organizers' hidden-test performance remains unknown.

## Reference

Malmsten, M., Börjeson, L., and Haffenden, C. (2020). *Playing with Words at the National Library of Sweden: Making a Swedish BERT*. [Paper](https://arxiv.org/abs/2007.01658)
