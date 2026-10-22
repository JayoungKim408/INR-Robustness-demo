MODEL_FLAGS="--data_name fashion --image_size 28 --in_channels 1 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --attention_resolutions 14,7"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"
CUDA_VISIBLE_DEVICES=7 python image_train.py --data_dir /nfs/stak/users/hongsa/INR-Robustness/raw $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS