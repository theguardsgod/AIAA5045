set -e
set -x
export PYTHONPATH='./src'

# Experient index!!
file_name=`basename $0`
experiment_index=${file_name##*_}
experiment_index=${experiment_index%%.*}


CUDA_VISIBLE_DEVICES=0 python -u src/trainer.py \
    --experiment_index=$experiment_index \
    --cudas=0 \
    --n_epochs=800 \
    --batch_size=96 \
    --server=lab_center \
    --eval_frequency=5 \
    --backbone=resnet50 \
    --learning_rate=1e-2 \
    --optimizer=Adam \
    --initialization=default \
    --num_classes=7 \
    --num_worker=4 \
    --input_channel=3 \
    2>&1 | tee $log_file
