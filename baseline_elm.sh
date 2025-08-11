cd lm-evaluation-harness
for model_pth in \
    /home/ubuntu/ELM/LMFlow/output_models/head1B_merged \
    /home/ubuntu/ELM/LMFlow/output_models/head1M_merged \
    /home/ubuntu/ELM/LMFlow/output_models/head2M_merged \
    /home/ubuntu/ELM/LMFlow/output_models/head10M_merged \
    /home/ubuntu/ELM/LMFlow/output_models/head100M_merged
do
    lm_eval --model hf \
        --model_args pretrained=$model_pth,trust_remote_code=True,cache_dir=/home/ubuntu/ELM/cache \
        --tasks elmb_roleplay,elmb_reasoning,elmb_functioncalling,elmb_chatrag \
        --device cuda:0 \
        --batch_size 1 \
        --log_samples \
        --output_path ./eval_results/test_elmb_$($model_pth)
done