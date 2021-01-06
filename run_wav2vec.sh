#!/bin/bash

#How to run?
python examples/wav2vec/wav2vec_manifest.py /home/azureuser/data/test_waves --dest /home/azureuser/data/waves_fairseq --ext wav --valid-percent 0.01


fairseq-hydra-train \
    task.data=/home/azureuser/data/waves_fairseq \
    distributed_training.distributed_world_size=4 +optimization.update_freq='[32]' \
    --config-dir examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech
   


python libri_labels.py /home/azureuser/data/waves_fairseq/train.tsv --output-dir /home/azureuser/data/labeled_data --output-name 'train'

python libri_labels.py /home/azureuser/data/waves_fairseq/valid.tsv --output-dir /home/azureuser/data/labeled_data --output-name 'valid'


/home/azureuser/data/fairseq/outputs/2021-01-05/12-38-10/checkpoints
checkpoint_best.pt  checkpoint_last.pt  checkpoint_last.pt.tmp

fairseq-hydra-train \
    task.data=/home/azureuser/data/labeled_data \
    model.w2v_path=/home/azureuser/data/fairseq/outputs/2021-01-05/12-38-10/checkpoints/checkpoint_best.pt \
    distributed_training.distributed_world_size=4 +optimization.update_freq='[6]' \
    --config-dir examples/wav2vec/config/finetuning \
    --config-name base_1h

/home/azureuser/data/fairseq/outputs/2021-01-05/14-56-31/checkpoints/checkpoint_last.pt
$subset=valid
python examples/speech_recognition/infer.py \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw \
--task audio_pretraining \
--nbest 1 \ 
--path /home/azureuser/data/fairseq/outputs/2021-01-05/14-56-31/checkpoints/checkpoint_last.pt \ 
--gen-subset $subset \
--results-path /path/to/save/results/for/sclite \
--w2l-decoder kenlm \
--lm-model /path/to/kenlm.bin \
--lm-weight 2 \
--word-score -1 \
--sil-weight 0 \
--criterion ctc \
--labels ltr \
--max-tokens 4000000 \
--post-process letter


#NOTE: to start from a checkpoint: add this param to fairseq-hydra-train
#--restore-file $SAVE_DIR/checkpoint_774_140000.pt 