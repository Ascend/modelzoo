python pwcnet_finetune_lg-6-2-multisteps-mpisintelclean.py
       --iterations 200000
       --display 1000 
       --save_path ./pwcnet-lg-6-2-multisteps-mpisintelclean-finetuned/
       --batch_size 4 
       --dataset ./dataset/
       --robust True
       --pretrained ./pretrained/pwcnet.ckpt-595000