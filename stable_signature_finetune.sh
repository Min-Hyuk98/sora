python stable_signature_finetune.py --num_keys 1 \
    --msg_decoder_path ./decoder/dec_48b_whit.torchscript.pt \
    --train_dir ./coco/train \
    --val_dir ./coco/val \
    --batch_size 1 \
    --img_size 256 \
    --lambda_i 0.2 \
    --loss_i watson-vgg \
    --loss_w bce \
    --optimizer AdamW,lr=5e-4 \
    --steps 100 \
    --warmup_steps 20 \
    --config configs/opensora-v1-2/train/stage1.py