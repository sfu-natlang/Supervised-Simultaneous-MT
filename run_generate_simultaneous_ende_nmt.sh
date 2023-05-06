#!/bin/bash

agent_path=agent5_trans_iwslt14_ende_big_nmt_bi_prepinput
data_path=/path/to/datafolder/data-bin/iwslt14.tokenized.de-en/
nmt_path=/path/to/model/folder/nmt_trans_iwslt14_ende_base/checkpoint_best.pt
gen_subset=test

src=en
trg=de
ref_file=$agent_path/test.word.de


for f in $agent_path/checkpoint1.pt; do
    CUDA_VISIBLE_DEVICES=""
    out_file=$( echo $f | cut -d "." -f 1).$gen_subset.txt
    if [ -f $out_file ]; then
        echo "$out_file already exists, skipping gereation process."
    else
        echo " processing $f"
        CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate_simultaneous.py $data_path -s $src -t $trg  --user-dir ../examples/Supervised_simul_MT --task Supervised_simultaneous_translation --gen-subset $gen_subset --path $nmt_path --left-pad-source False --max-tokens 4000 --skip-invalid-size-inputs-valid-test --has-target False --beam 5 --agent-path $f > $out_file
    fi

    action_file=$(echo $out_file | cut -d "." -f 1).$gen_subset.actions
    hypos_file=$(echo $out_file | cut -d "." -f 1).$gen_subset.hypos
    
    if [ -f $action_file ]; then
        echo "$action_file already exists, skipping gereation process."
    else
        echo "generating action file"
        python scripts/sort-actions.py $out_file 5 > $action_file
        sed "y/45/01/" $action_file > $action_file.01
        rm $action_file
        mv $action_file.01 $action_file
    fi

    python $agent_path/STACL/metricAL.py $action_file

    if [ -f $hypos_file ]; then
        echo "$hypos_file already exists, skipping gereation process."
    else
        echo "generating hypothesis file"
        python scripts/sort-hypotheses.py $out_file 5 > $hypos_file
        sed -e "s/@@ //g" $hypos_file > $hypos_file.words 
    fi

    /cs/natlang-data/WMT19/News/mosesdecoder/scripts/generic/multi-bleu.perl $ref_file < $hypos_file.words

done

