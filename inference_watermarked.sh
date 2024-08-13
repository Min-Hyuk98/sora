# PROMPTS=(A_dog_in_astronaut_suit_and_sunglasses_floating_in_space An_epic_tornado_attacking_above_aglowing_city_at_night Slow_pan_upward_of_blazing_oak_fire_in_an_indoor_fireplace Sunset_over_the_sea Yellow_and_black_tropical_fish_dart_through_the_sea a_cat_wearing_sunglasses_and_working_as_a_lifeguard_at_pool) #waterf
PROMPTS=(a_beautiful_waterfall)
num_frames=(2s)
resolution=(480p)
aspect_ratio=(1:1)
fps=(16)




for PROMPT in "${PROMPTS[@]}"
  do
    python scripts/inference_watermark1.py --config configs/opensora-v1-2/inference/sample.py \
    --num-frames ${num_frames} --resolution ${resolution} --aspect-ratio ${aspect_ratio} --fps ${fps} \
    --prompt ${PROMPT} --watermarked_eval --watermarked_vae_ckpt ./output/checkpoint_000.pth


  python scripts/inference_watermark2.py --config configs/opensora-v1-2/inference/sample.py \
    --num-frames ${num_frames} --resolution ${resolution} --aspect-ratio ${aspect_ratio} --fps ${fps} \
    --output_save_dir ./samples_w/${PROMPT} \
    --prompt ${PROMPT} --watermarked_eval --watermarked_vae_ckpt ./output/checkpoint_000.pth
  done
