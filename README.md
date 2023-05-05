# Translation-based Supervision for Policy Generation in Simultaneous Neural Machine Translation

<img src="./oracle.png" width="600" class="center">


# Table of contents
1. [Installation](#installation)
2. [Getting Started](#Getting-Started)
3. [Citation](#citation)

# Installation
You may need to run `pip install --editable .` first.

# Getting Started
Here is the simplest way to train agent on IWSLT14 dataset for DE -> EN language direction:

## Generating oracle action sequences
In order to generate action sequences for Test set run the the following command:

```
python fairseq_cli/generate_action.py /path/to/datafolder/data-bin/iwslt14.tokenized.de-en/
-s de
-t en
--user-dir ../examples/Supervised_simul_MT
--task Supervised_simultaneous_translation
--gen-subset test
--path /path/to/model/folder/nmt_trans_iwslt14_ende_base/checkpoint_best.pt
--left-pad-source False
--max-tokens 8000
--beam 10 > /path/to/model/folder/nmt_trans_iwslt14_ende_base/action_sequence.txt
```
The directory `/path/to/datafolder/data-bin/iwslt14.tokenized.de-en/` refers to the bin folder of modified data by fairseq, where the .bin and .idx files are located. You may need to add an additional dictionary for the actions. There is a sample action dictionary in this repo, called: `dict.act.txt`. Place this in the same data folder next next to the other data files.
Then you can run:

```
python scripts/sort-actions.py /path/to/model/folder/nmt_trans_iwslt14_ende_base/action_sequence.txt 5 > /path/to/model/folder/nmt_trans_iwslt14_ende_base/action_sequence.lines.txt
```

to have a clean folder of action sequences for each sentence. The `sort-source.py` and `sort-target.py` also extracts the source and target sentences respectively.

## Training the model

Before starting to train the model, for each sentence in our dataset we need to have a gold action sequence and the translation generated by the MT model following the gold actions. We can generate such input by running the following command on our train, valid, and test subsets:

```
python fairseq_cli/generate_input.py /path/to/datafolder/iwslt14.tokenized.de-en/bin-en_de/
-s en -t de
--user-dir ../examples/Supervised_simul_MT
--task Supervised_simultaneous_translation
--gen-subset test
--path /path/to/model/folder/nmt_trans_iwslt14_ende_base/checkpoint_best.pt
--left-pad-source False
--max-tokens 8000
--skip-invalid-size-inputs-valid-test
--beam 5
--has-target False > /path/to/model/folder/nmt_trans_iwslt14_ende_base/test.beam5_notarget.input.txt
```
Repeat above command for the train, valid, and test subsets and then run `run_generate_input.sh` to generate appropriate inputs for training. Please double check the value for `mt_path` and `data_folder` in the `run_generate_input.sh` to make sure the script is running on the right directory.\
\
The last step to preprocess the data before training is to run:
```
python preprocess.py -s src -t trg -a act 
--trainpref data_input/nmt_trans_iwslt14_ende_base/train.beam5_notarget.input.txt
—validpref data_input/nmt_trans_iwslt14_ende_base/valid.beam5_notarget.input.txt
—testpref data_input/nmt_trans_iwslt14_ende_base/test.beam5_notarget.input.txt
--tgtdict data_input/nmt_trans_iwslt14_ende_base/dict.trg.txt 
--srcdict data_input/nmt_trans_iwslt14_ende_base/dict.src.txt 
--actdict data_input/nmt_trans_iwslt14_ende_base/dict.act.txt
--destdir data_input/nmt_trans_iwslt14_ende_base/bin-data
```
Before running this you should copy the dictionaries from the bin folder of the Fairseq data directory `/path/to/datafolder/data-bin/iwslt14.tokenized.de-en/` and change the name of the `dict.en.txt` and `dict.de.txt` to `dict.src.txt` and `dict.trg.txt` names respectively. This way we can ensure we'll use the same dictionary as the NMT model.
\
Then we can pass these files to start training the model:

```
python train.py data_input/nmt_trans_iwslt14_ende_base/bin-data
-s src -t trg
--clip-norm 5
--save-dir ./agent5_trans_iwslt14_ende_big_nmt_bi_prepinput/
--user-dir ../examples/Supervised_simul_MT
--max-epoch 50
--lr 8e-4
--dropout 0.4
--lr-scheduler fixed
--optimizer adam
--arch agent_lstm_5_big
--task Supervised_simultaneous_translation
--update-freq 4
--skip-invalid-size-inputs-valid-test
--sentence-avg
--criterion label_smoothed_cross_entropy
--label-smoothing 0.1
--batch-size 100
--ddp-backend=no_c10d
--nmt-path /path/to/model/folder/nmt_trans_iwslt14_ende_base/checkpoint_best.pt
--report-accuracy
--lr-shrink 0.95
--force-anneal 4
--has-reference-action | tee ./agent5_trans_iwslt14_ende_big_nmt_bi_prepinput/agent5_trans_iwslt14_ende_big_nmt_bi_prepinput.log
```

## Testing the model

After finishing the training process, we can use the `run_generate_simultaneous_ende_nmt.sh` to generate the actions and translations simultaneously, and receive the evaluation metrics. Please open the script and change the values for `agent_path`, `data_path` \[The path to the Fairseq data bin directory\] , `nmt_path`, and `ref_file` \[The path to the reference file that you want to use as the gold translations.\] according to your local directories.

## Liscence

Main parts of the code is coming from https://github.com/facebookresearch/fairseq. All its liscenes also applies to this repo.

# Citation
```
@inproceedings{alinejad-etal-2021-translation,
    title = "Translation-based Supervision for Policy Generation in Simultaneous Neural Machine Translation",
    author = "Alinejad, Ashkan  and
      Shavarani, Hassan S.  and
      Sarkar, Anoop",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.130",
    pages = "1734--1744",
}
```
