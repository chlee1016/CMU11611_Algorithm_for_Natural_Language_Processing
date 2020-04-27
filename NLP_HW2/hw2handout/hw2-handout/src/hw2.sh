#!/bin/bash
# Sample script file to run your code. Feel free to change it.
# Run this script using ./hw2.sh train_text train_label test_text test_label
# Example:  ./hw2.sh ../dev_text.txt ../dev_label.txt ../heldout_text.txt ../heldout_pred_nb.txt
# $ sh hw2.sh ../dev_text.txt ../dev_label.txt ../heldout_text.txt ../heldout_pred_nb.txt

echo "Running using train file at" $1 $'\n' "and the label at" $2 $'\n' \
"and test file at" $3 $'\n' "and save pred at" $4 $'\n'
python naivebayes.py $1 $2 $3 $4


