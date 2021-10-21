#!/bin/bash

mt_path=nmt_trans_iwslt14_deen_base
data_folder=data_input

mkdir $data_folder/$mt_path

for f in $mt_path/*input.txt; do

    out_dest=$data_folder/$mt_path

    act_file=$( echo $f | cut -d "." -f -4).act
    src_file=$( echo $f | cut -d "." -f -4).src
    trg_file=$( echo $f | cut -d "." -f -4).trg

    if [ -f $act_file ]; then
        echo "$act_file already exists, skipping gereation process."
    else
        echo "generaating $act_file"
        python scripts/sort-input-actions.py $f 5 > $act_file
        sed "y/45/01/" $act_file > $act_file.01
        rm $act_file
        mv $act_file.01 $act_file
    fi

    if [ -f $src_file ]; then
        echo "$src_file already exists, skipping gereation process."
    else
        echo "generaating $src_file"
        python scripts/sort-source.py  $f 5 > $src_file
    fi

    if [ -f $trg_file ]; then
        echo "$trg_file already exists, skipping gereation process."
    else
        echo "generaating $trg_file"
        python scripts/sort-target.py  $f 5 > $trg_file
    fi

    scp $act_file $out_dest
    scp $src_file $out_dest
    scp $trg_file $out_dest
done
