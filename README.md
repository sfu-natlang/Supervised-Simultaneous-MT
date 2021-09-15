# Translation-based Supervision for Policy Generation in Simultaneous Neural Machine Translation

<img src="./oracle.png" width="600" class="center">

You may need to run `pip install --editable .` first.

In order to generate action sequences for Test set run the the following command:

```
python fairseq_cli/generate_action.py /path/to/datafolder/data-bin/wmt14_en_de/
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
Then you can run

```
python scripts/sort-sentences.py /path/to/model/folder/nmt_trans_wmt14_deen_med/action_sequence.txt 5 > /path/to/model/folder/nmt_trans_wmt14_deen_med/action_sequence.lines.txt
```

to have a clean folder of action sequences for each sentence.

## Quick start
Here is the simplest way to train agent on IWSLT14 dataset for DE -> EN language direction:

### Generating oracle action sequences

```
python fairseq_cli/generate_input.py ./../fairseq/examples/translation/iwslt14.tokenized.de-en/bin-en_de/  -s de -t en --user-dir ../examples/Supervised_simul_MT --task Supervised_simultaneous_translation --gen-subset test --path ./multipath_iwslt14_de_en/checkpoint_best.pt --left-pad-source False --max-tokens 10000 --skip-invalid-size-inputs-valid-test --beam 5 --has-target False --eval-waitk 5 > ./multipath_iwslt14_de_en/test.beam5_notarget.input.txt
```

# Citation

TBD
