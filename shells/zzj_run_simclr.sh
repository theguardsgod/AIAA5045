cd /home/zhuzhengjie/root/CODE/AIAA5045/SimCLR

python run.py \
-data /data/zhuzhengjie/DATA/ISIC2018/fk_subfolder \
--arch densenet121 \
--dataset_name isic2018 \
--log-every-n-steps 100 \
--epochs 100 \
--gpu_index 2 \
--batch_size 84 \
--seed 32 \
