#!/bin/bash
source path.sh

python3 main3_ddp.py \
        --BATCH_SIZE 32 \
        --N_EPOCHS 300 \
        --save_dir /home3/hexin/AcademiCodec/logs \
        --PATH  /home3/hexin/AcademiCodec/saved_models \
        --train_data_path /home3/hexin/speechbrain/recipes/VoxLingua107/lang_id/fleurs_train.json \
        --valid_data_path /home3/hexin/speechbrain/recipes/VoxLingua107/lang_id/fleurs_valid.json \
        --sr 16000 \
        --ratios 6 5 4 2 \
        --target_bandwidths 1 2 4 8 12
