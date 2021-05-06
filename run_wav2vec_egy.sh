#!/bin/bash

#########################################################################################################
# 1. Prepare training data manifest:
src_path="/home/maggie/data_ext/audio_data_unzipped/train_data_10_25ms"
dst_path="/home/maggie/data_ext/wav2vec_manifest"

python examples/wav2vec/wav2vec_manifest_egy.py $src_path --dest $dst_path --ext "wav" --valid-percent 0.01
#########################################################################################################

#Train a wav2vec 2.0 base model:
n_gpus_required=128
n_gpus_actual=8
x=$(( $n_gpus_required/$n_gpus_actual ))

data_path_pretrain="/home/maggie/data_ext/wav2vec_manifest"
config_dir_pretrain="examples/wav2vec/config/pretraining"
config_name_pretrain="wav2vec2_large_librivox"

#using xlsr:
chpt='/home/maggie/data_ext/xlsr_53_56k.pt'

fairseq-hydra-train task.data=$data_path_pretrain \
    checkpoint.restore_file=$chpt checkpoint.reset_dataloader='true'\
    checkpoint.reset_lr_scheduler='true' checkpoint.reset_optimizer='true' checkpoint.reset_meters='true' \
    distributed_training.distributed_world_size=$n_gpus_actual +optimization.update_freq="[$x]" \
    --config-dir $config_dir_pretrain --config-name $config_name_pretrain

#from scratch:
fairseq-hydra-train task.data=$data_path_pretrain \
    distributed_training.distributed_world_size=$n_gpus_actual +optimization.update_freq="[$x]" \
    --config-dir $config_dir_pretrain --config-name $config_name_pretrain

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

#Fine-tune a pre-trained model with CTC:

#generate manifest files for labelled data
# TODO : ############################################ 

#generate .wrd and .ltr and dict files using this script:
train_tsv="/home/maggie/data_ext/wav2letter_manifest/train.tsv"
valid_tsv="/home/maggie/data_ext/wav2letter_manifest/valid.tsv"

output_dir="/home/maggie/data_ext/wav2letter_manifest"

python libri_labels.py $train_tsv --output-dir $output_dir --output-name 'train'
python libri_labels.py $valid_tsv --output-dir $output_dir --output-name 'valid'

#Fine-tuning with letter targets:
data_path_finetune="/home/azureuser/data/waves_fairseq"
w2v_model="/path/to/model.pt"
config_dir_finetune="examples/wav2vec/config/finetuning"
config_name_finetune="base_egy_30h"

fairseq-hydra-train task.data=$data_path_finetune model.w2v_path=$w2v_model \
    --config-dir $config_dir_finetune --config-name $config_name_finetune

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

#Evaluating a CTC model:

model="/path/to/model"
results_path="/path/to/save/results/for/sclite"
kenlm_path="/path/to/kenlm.bin"

$subset="valid"
python examples/speech_recognition/infer.py /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw \
--task audio_pretraining \
--nbest 1 
--path $model \
--gen-subset $subset \ 
--results-path $results_path \
--w2l-decoder kenlm \
--lm-model $kenlm_path \
--lm-weight 2 \
--word-score -1 \
--sil-weight 0 \
--criterion ctc \
--labels ltr \
--max-tokens 4000000 \
--post-process letter
