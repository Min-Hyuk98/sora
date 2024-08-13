# PROMPTS=(A_dog_in_astronaut_suit_and_sunglasses_floating_in_space An_epic_tornado_attacking_above_aglowing_city_at_night Slow_pan_upward_of_blazing_oak_fire_in_an_indoor_fireplace Sunset_over_the_sea Yellow_and_black_tropical_fish_dart_through_the_sea a_cat_wearing_sunglasses_and_working_as_a_lifeguard_at_pool) #waterf
PROMPTS=(a_beautiful_waterfall)
for PROMPT in "${PROMPTS[@]}"
    do
        python metrics_stable_signature.py --eval_imgs True --eval_bits True --output_data_name ${PROMPT}\
        --img_dir ./samples_w/${PROMPT} --img_dir_nw ./samples_nw/${PROMPT}
    done



