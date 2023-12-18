cd /home/zhuzhengjie/root/CODE/AIAA5045/SimCLR

python run.py \
-data /data/zhuzhengjie/DATA/ISIC2018/fk_subfolder \
--arch resnet50 \
--dataset_name isic2018 \
--log-every-n-steps 100 \
--epochs 100 \
--gpu_index 6 \
--batch_size 32 \
--seed 32 \
