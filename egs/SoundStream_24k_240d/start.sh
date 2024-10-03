#!/bin/bash
source path.sh

python3 main3_ddp.py \
        --BATCH_SIZE 16 \
        --N_EPOCHS 300 \
        --save_dir /home/users/ntu/ccdshyzh/AcademiCodec/logs \
        --PATH  /home/users/ntu/ccdshyzh/AcademiCodec/saved_models \
        # --train_data_path /home3/hexin/speechbrain/recipes/VoxLingua107/lang_id/fleurs_train.json \
        # --valid_data_path /home3/hexin/speechbrain/recipes/VoxLingua107/lang_id/fleurs_valid.json \
        --train_data_path /home/users/ntu/ccdshyzh/datasets/LibriSpeech/train-clean-100/103/1240 \
        --valid_data_path /home/users/ntu/ccdshyzh/datasets/LibriSpeech/train-clean-100/103/1241 \
        --sr 16000 \
        --ratios 6 5 4 2 \
        --target_bandwidths 1 2 4 8 12
