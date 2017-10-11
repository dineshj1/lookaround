# First train with T=1
th lookaround.lua  --jobno 1 --cuda --no_loadInit --full_data +1 --noTest 1 --initModel NA --learningRate 40 --lr_style constant --finetuneDecoderFlag 1 --rewardScale 0.01 --rho 1 --maxTries 10000 --maxEpoch 10000 --manual_seed 38 --action_gridsize_factor 0.6
# Then train with T=4, initializing with T=1 model
th lookaround.lua --jobno 2 --cuda --no_loadInit --full_data +1 --noTest 1 --initModel ./outputs/models/1.dat --replaceActor 1 --learningRate 15 --lr_style constant --finetune_lrMult 0 --finetuneDecoderFlag 1 --rewardScale 0.01 --rho 4 --maxTries 10000 --maxEpoch 10000 --manual_seed 38 --action_gridsize_factor 0.6
