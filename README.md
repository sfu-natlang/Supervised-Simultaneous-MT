# Supervised Simultaneous Machine Translation

In order to generate action sequences for Test set run the the following command:

```
python fairseq_cli/generate_action.py /path/to/datafolder/data-bin/wmt14_en_de/bin-de_en-encoded
-s de
-t en
--user-dir ../examples/Supervised_simul_MT
--task Supervised_simultaneous_translation
--gen-subset test
--path /path/to/model/folder/nmt_trans_wmt14_deen_med/checkpoint_best.pt
--left-pad-source False
--max-tokens 8000
--beam 10 > /path/to/model/folder/nmt_trans_wmt14_deen_med/action_sequence.txt
```
