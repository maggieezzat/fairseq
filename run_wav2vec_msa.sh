#!/bin/bash

# Training a new model with the CLI tools

#NOTE: to start from a checkpoint: add this param to fairseq-hydra-train
#--restore-file $SAVE_DIR/checkpoint_774_140000.pt 

#########################################################################################################
# 1. Prepare training data manifest:
#Given a directory containing wav files to be used for pretraining (we recommend splitting each file into separate file 10 to 30 seconds in length)

ext="wav"  #ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.
valid=0.01 #valid should be set to some reasonable percentage (like 0.01) of training data to use for validation.

src_path="/home/azureuser/data/audio_data_processed/modern-standard-arabic"
dst_path="/home/azureuser/data/wav2vec_data/modern-standard-arabic"

python examples/wav2vec/wav2vec_manifest_egy.py $src_path --dest $dst_path --ext $ext --valid-percent $valid
#########################################################################################################

# 2. Train a wav2vec 2.0 base model:

#NOTE: that the input is expected to be single channel, sampled at 16 kHz
#NOTE: you can simulate 64 GPUs by using k GPUs and adding command line parameters (before --config-dir) 
#distributed_training.distributed_world_size=k +optimization.update_freq='[x]' where x = 64/k

n_gpus_required=64
n_gpus_actual=8
x=$(( $n_gpus_required/$n_gpus_actual ))

data_path_pretrain="/home/maggie/data/wav2vec_data/modern-standard-arabic"

config_dir_pretrain="examples/wav2vec/config/pretraining"
config_name_pretrain="wav2vec2_base_egy_data"

fairseq-hydra-train task.data=$data_path_pretrain \
    distributed_training.distributed_world_size=$n_gpus_actual +optimization.update_freq="[$x]" \
    --config-dir $config_dir_pretrain --config-name $config_name_pretrain

### FINETUNE WAV2VEC2 LARGE

n_gpus_required=128
n_gpus_actual=8
x=$(( $n_gpus_required/$n_gpus_actual ))
data_path_pretrain="/home/maggie/data/new_msa_data/w2v_manifest_dir"
config_dir_pretrain="examples/wav2vec/config/pretraining"
config_name_pretrain="wav2vec2_large_librivox"
chpt="/home/maggie/data/xlsr_53_56k.pt"

fairseq-hydra-train task.data=$data_path_pretrain \
    checkpoint.restore_file=$chpt \
    distributed_training.distributed_world_size=$n_gpus_actual +optimization.update_freq="[$x]" \
    --config-dir $config_dir_pretrain --config-name $config_name_pretrain

#########################################################################################################

# 3. Fine-tune a pre-trained model with CTC:

#Prepare training data manifest:
ext="wav" 

src_path="/home/maggie/data/audio_data_labelled/modern-standard-arabic/common_voice_ar/train"
dst_path="/home/maggie/data/wav2letter_data/modern-standard-arabic"
python examples/wav2vec/wav2vec_manifest_egy.py $src_path --dest $dst_path --ext $ext --valid-percent 0.0

src_path="/home/maggie/data/audio_data_labelled/modern-standard-arabic/common_voice_ar/dev"
dst_path="/home/maggie/data/wav2letter_data/modern-standard-arabic"
python examples/wav2vec/wav2vec_manifest_egy.py $src_path --dest $dst_path --ext $ext --valid-percent 0.0

src_path="/home/maggie/data/audio_data_labelled/modern-standard-arabic/common_voice_ar/test"
dst_path="/home/maggie/data/wav2letter_data/modern-standard-arabic"
python examples/wav2vec/wav2vec_manifest_egy.py $src_path --dest $dst_path --ext $ext --valid-percent 0.0



#Fine-tuning a model requires parallel audio and labels file, as well as a vocabulary file in fairseq format. 
#generate .wrd and .ltr and dict files using this script:
train_tsv="/home/maggie/data/wav2letter_data/modern-standard-arabic/train.tsv"
train_trans="/home/maggie/data/audio_data_labelled/modern-standard-arabic/common_voice_ar/train_trans.txt"

valid_tsv="/home/maggie/data/wav2letter_data/modern-standard-arabic/valid.tsv"
valid_trans="/home/maggie/data/audio_data_labelled/modern-standard-arabic/common_voice_ar/dev_trans.txt"

test_tsv="/home/maggie/data/wav2letter_data/modern-standard-arabic/test.tsv"
test_trans="/home/maggie/data/audio_data_labelled/modern-standard-arabic/common_voice_ar/test_trans.txt"

output_dir="/home/maggie/data/wav2letter_data/modern-standard-arabic"

python examples/wav2vec/msa_labels.py $train_tsv $train_trans --output-dir $output_dir --output-name 'train'
python examples/wav2vec/msa_labels.py $valid_tsv $valid_trans --output-dir $output_dir --output-name 'valid'
python examples/wav2vec/msa_labels.py $test_tsv $test_trans --output-dir $output_dir --output-name 'test'

#Fine-tuning with letter targets:
data_path_finetune="/home/azureuser/data/waves_fairseq"
w2v_model="/path/to/model.pt"
config_dir_finetune="examples/wav2vec/config/finetuning"
config_name_finetune="base_msa_16h"

#Note: you can simulate 24 GPUs by using k GPUs and adding command line parameters (before --config-dir) distributed_training.distributed_world_size=k +optimization.update_freq='[x]' where x = 24/k

fairseq-hydra-train task.data=$data_path_finetune model.w2v_path=$w2v_model \
    --config-dir $config_dir_finetune --config-name $config_name_finetune

#NOTE: Decoding with a language model during training requires wav2letter python bindings. 
#If you want to use a language model, add +criterion.wer_args='[/path/to/kenlm, /path/to/lexicon, 2, -1]' to the command line.
#tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)
#wer_args:DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)

#wer_kenlm_model: if this is provided, use kenlm to compute wer (along with other wer_* args)
#wer_lexicon: lexicon to use with wer_kenlm_model
#wer_lm_weight: default=2.0, lm weight to use with wer_kenlm_model
#wer_word_score: default=-1.0, lm word score to use with wer_kenlm_model


#########################################################################################################

#Evaluating a CTC model:

#Evaluating a CTC model with a language model requires wav2letter python bindings to be installed.
#Fairseq transformer language model used in the wav2vec 2.0 paper can be obtained from the wav2letter model repository. 
#Be sure to upper-case the language model vocab after downloading it.
#Letter dictionary for pre-trained models can be found here.
#Next, run the evaluation command:

#data to train wav2letter on
data="/home/maggie/data/wav2letter_data/modern-standard-arabic"
#w2v model
model="/home/maggie/data/fairseq/outputs"
#path ro store results in
results_path="/home/maggie/data/msa_w2l_results"
#binary lm path
kenlm_path="/home/maggie/data/lm_data/modern-standard-arabic/msa_lm.binary"
#subset of data to evaluate on
$subset="test"

python examples/speech_recognition/infer.py $data \
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
