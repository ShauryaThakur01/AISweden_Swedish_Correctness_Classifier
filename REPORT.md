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

Each candidate had 124,691,713 parameters and trained on 385 examples. Across the same 93 validation texts, the supplied-data model ranked texts more accurately and more than halved the mean absolute error. The generated-data model still showed some transfer to the assignment texts (ROC-AUC 0.797), but the supplied-data model led on all three metrics. Its higher Spearman correlation determined selection for the public test.

On the 120 public test examples, the selected model achieved **0.832 Spearman and 0.953 ROC-AUC**. The evaluator recorded **12.3 seconds and 586 samples per minute**, including model loading. The run used Windows and PyTorch 2.6.0 on an AMD CPU (family 25, model 80). The assignment reports 0.528 Spearman, 0.783 ROC-AUC and 714 samples per minute for its starter baseline on this split. KB-BERT scored higher on both quality measures. The baseline used a separately reported laptop CPU, so the timings cannot isolate the effect of model choice.

## Limitations and error analysis

**The selected model gave overly generous scores to poor texts.** On the 47 validation examples labeled poor, its mean prediction was 0.489 against a mean reference score of 0.335. On the 46 good examples, its mean prediction of 0.946 was close to the reference mean of 0.958. A held-out sentence pair shows how a high overall ROC-AUC can coexist with a missed grammatical distinction. In a sentence about a project meeting, changing *vad som återstår* to *vad som återstå* removes the finite verb ending. The reference score fell from 0.95 to 0.35, while the model's prediction moved only from 0.954 to 0.947.

**The generated-data model drew a weaker distinction between good and poor texts.** Its mean predictions were 0.793 for good examples and 0.643 for poor ones. One revealing case was *Hon köpte en ny vinter jacka.* The split compound should be written *vinterjacka*. Against a reference of 0.45, the generated-data model predicted 0.879 and the supplied-data model predicted 0.473. The generator never introduces split compounds. It also uses only three sentence frames and assigns a fixed score to each error count. Since sentence variety, error coverage and scoring rules all changed together, this experiment cannot determine which contributed most to the performance gap.

**Scope of the comparison.** The result supports choosing the supplied examples over this particular generator at a budget of 385 training texts. Both validation and public test texts come from the assignment, and only the selected model was evaluated on the public test. A single grouped split and training seed leave the variability of the validation gap unmeasured. Whether synthetic examples help when added to the supplied data, or whether these scores transfer to naturally occurring learner writing, remains open.

## Reference

Malmsten, M., Börjeson, L., and Haffenden, C. (2020). *Playing with Words at the National Library of Sweden: Making a Swedish BERT*. [Paper](https://arxiv.org/abs/2007.01658)
