# cs285finalproject
# cs285project

# to run fcn: 

uv run modal run --detach src/scripts/modal_run.py \
    --env-name Humanoid-v5 \
    --base-config sacbc \
    --mode offline \
    --minari-dataset mujoco/humanoid/expert-v0 \
    --training-steps 500000 \
    --seed 0 \
    --run-group DivLearn-FCN \
    --encoder-type fcn \
    --kappa [kappa] \
    --dl-base-lr [encoder learning rate] \
    --dl-lr-cap [max encoder learning rate] \
    --bc-ramp-scale [bc-ramp-scale] \
    --encoder-kwargs '{"hidden_sizes":[JUST ENTER HIDDEN DIMS AS A LIST],"out_dim":[OUT DIM]}'

    
`dl-lr-cap` is a maximum value on the encoder learning rate, just because we are dividing and clipping by kappa but we don't want values to get too high   
`bc_ramp_scale` is just over how many steps you want the encoder weight in the BC loss to ramp up, higher values will mean more steps means slower 


# to run transformer 
uv run modal run --detach src/scripts/modal_run.py \
    --env-name Humanoid-v5 \
    --base-config sacbc \
    --mode offline \
    --minari-dataset mujoco/humanoid/expert-v0 \
    --training-steps 500000 \
    --seed 0 \
    --run-group DivLearn-FCN \
    --encoder-type transformer \
    --kappa [kappa] \
    --dl-base-lr [encoder learning rate] \
    --dl-lr-cap [max encoder learning rate] \
    --bc-ramp-scale [bc-ramp-scale]\
    --encoder-kwargs  '{"d_model":[D_MODEL],"out_dim":[OUT DIM],"num_layers":[NUM_LAYERS],"num_heads":[NUM_HEADS]}' 
