#!/bin/bash

# Training a new model with the CLI tools

#NOTE: to start from a checkpoint: add this param to fairseq-hydra-train
#--restore-file $SAVE_DIR/checkpoint_774_140000.pt 

#########################################################################################################
# 1. Prepare training data manifest:
#Given a directory containing wav files to be used for pretraining 
#(we recommend splitting each file into separate file 10 to 30 seconds in length)

ext="wav"  #ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.
valid=0.01 #valid should be set to some reasonable percentage (like 0.01) of training data to use for validation.

src_path="/home/azureuser/data/test_waves"
dst_path="/home/azureuser/data/waves_fairseq"

python examples/wav2vec/wav2vec_manifest.py $src_path --dest $dst_path --ext $ext --valid-percent $valid
#########################################################################################################

#Train a wav2vec 2.0 base model:

#This configuration was used for the base model trained on the Librispeech dataset in the wav2vec 2.0 paper
#NOTE: that the input is expected to be single channel, sampled at 16 kHz

#Note: you can simulate 64 GPUs by using k GPUs and adding command line parameters (before --config-dir) 
#distributed_training.distributed_world_size=k +optimization.update_freq='[x]' where x = 64/k

n_gpus_required=64
n_gpus_actual=24
x=$(( $n_gpus_required/$n_gpus_actual ))

data_path_pretrain="/home/azureuser/data/waves_fairseq"

config_dir_pretrain="examples/wav2vec/config/pretraining"
config_name_pretrain="wav2vec2_base_egy_data"

fairseq-hydra-train task.data=$data_path_pretrain \
    distributed_training.distributed_world_size=$n_gpus_actual +optimization.update_freq="[$x]" \
    --config-dir $config_dir_pretrain --config-name $config_name_pretrain
#########################################################################################################

#Fine-tune a pre-trained model with CTC:
#Fine-tuning a model requires parallel audio and labels file, as well as a vocabulary file in fairseq format. 

#generate .wrd and .ltr and dict files using this script:
train_tsv="/home/azureuser/data/waves_fairseq/train.tsv"
valid_tsv="/home/azureuser/data/waves_fairseq/valid.tsv"

output_dir="/home/azureuser/data/labeled_data"

python libri_labels.py $train_tsv --output-dir $output_dir --output-name 'train'
python libri_labels.py $valid_tsv --output-dir $output_dir --output-name 'valid'

#Fine-tuning with letter targets:
data_path_finetune="/home/azureuser/data/waves_fairseq"
w2v_model="/path/to/model.pt"
config_dir_finetune="examples/wav2vec/config/finetuning"
config_name_finetune="base_egy_30h"

#Note: you can simulate 24 GPUs by using k GPUs and adding command line parameters (before --config-dir) distributed_training.distributed_world_size=k +optimization.update_freq='[x]' where x = 24/k

fairseq-hydra-train task.data=$data_path_finetune model.w2v_path=$w2v_model \
    --config-dir $config_dir_finetune --config-name $config_name_finetune

#NOTE: Decoding with a language model during training requires wav2letter python bindings. 
#If you want to use a language model, add +criterion.wer_args='[/path/to/kenlm, /path/to/lexicon, 2, -1]' to the command line.
#########################################################################################################

#Evaluating a CTC model:

#Evaluating a CTC model with a language model requires wav2letter python bindings to be installed.
#Fairseq transformer language model used in the wav2vec 2.0 paper can be obtained from the wav2letter model repository. 
#Be sure to upper-case the language model vocab after downloading it.
#Letter dictionary for pre-trained models can be found here.
#Next, run the evaluation command:

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
