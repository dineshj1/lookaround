require 'dp'
require 'nn'
require 'rnn'
require 'os'
require 'optim'
require 'lfs'
cwd=lfs.currentdir();
package.path = package.path .. ';' .. cwd .. '/?.lua;'
require ('SUN360ActiveMod')
require ('IdentityFeedback')
require ('VRRegressReward')
require ('MSEFeedback')
require ('SaveModelToFile')
require ('CircShift')
require ('OneShotRNN')
require ('DummyContainer')
require ('actorMod')

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf

io.output():setvbuf('no');
print "Running on:"
os.execute('hostname'); -- announces which machine it is running on

version = 10

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Recurrent Model for Visual Attention')
cmd:text('Example:')
cmd:text('$> th active_reconstruct.lua > results.txt')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--finalLR_ratio', 1, 'final learning rate/ starting learning rate')
--cmd:option('--cyclicalLRFlag', 0, 'if 1, learning rate goes from opt.learningRate to opt.finalLR in opt.saturateEpoch iterations, then back again, and so on.')
cmd:option('--saturateEpoch', 1000, 'epoch at which linear decayed LR will reach finalLR')
cmd:option('--lr_style', 'step_if_stagnant', 'how to get from origLR to finalLR - constant | linear | step | cyclical | exp | step_exp | step_repeat | step_if_stagnant')
cmd:option('--stagnantWindow', 200, 'window over which stagnancy is measured (only for step_if_stagnant)')
cmd:option('--stagnantDelta', 0.0002, 'minimum amount that best training loss should have changed (only_for_step_if_stagnant)')
cmd:option('--step_decayFactor', 0.5, 'earlier step_exp_decayFactor. for step_exp lr_style, learning rate decay factor after reaching saturateEpochs. 0.9931 corresponds to halving every 100 epochs. 0.9986 - halving every 500 epochs. 0.9993- halving every 1000 epochs. for step_repeat lr_style, it is the gamma factor by which learning rate is dropped every saturateEpochs epochs.')
cmd:option('--momentum', 0.9, 'momentum')

cmd:option('--explorationBaseFactor', 0);
cmd:option('--explorationDecay', 0);
cmd:option('--wrapAroundRows', 1, 'whether rows wrap around or not (azimuths do)');
cmd:option('--wrapAroundCols', -1, 'whether columns wrap around or not (elevations usually dont)');

cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', -1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 2000, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--reconstructionCriterion', 'MSECriterion', 'which criterion to use for reconstruction')
cmd:option('--aggregateReconstructions', 'off', 'how to add reconstructions over timesteps: off | sum | avg | learned_sum')
cmd:option('--memorizeObservedViews', 1, 'whether to directly copy observed views into appropriate position in target viewgrid')
cmd:option('--initStyle', 'uniform', 'initialization method: uniform | xavier');
cmd:option('--initLim', 0.1, 'for uniform initStyle, parameters are initialized to be between -initLim and +initLim.');
cmd:option('--weightDecay', 0.005, 'how much parameters should decay per iteration... basically L2 regularization')
cmd:option('--biasDecayFlag', 1, 'whether weightDecay should also be applied to biases')
cmd:option('--featDropout', 0, 'dropout applied straight to inputs to combat overfitting')
cmd:option('--combineDropout', 0, 'dropout applied to output of combine module to avoid overfitting')
cmd:option('--batchNormFlag', 1, 'whether or not to introduce batch normalization at various points in the network') -- on by default
cmd:option('--initFeedbackFlag', 1, 'whether to initialize recurrent feedback module')
cmd:option('--initFeedbackType', 'identity', 'how to initialize recurrent module -- can be "identity" or "zero"')
cmd:option('--aggregatorStyle', 'lstm', 'rnn | lstm');
cmd:option('--noMemory', -1, 'if >0, RNN feedback module zeros out, so the RNN behaves like the exact same feedforward module at each time step');
cmd:option('--identFeedbackFlag', -1, 'whether to keep recurrent feedback module at pure identity always')
cmd:option('--avgAtTestFlag', -1, 'whether to keep recurrent feedback module at identity always')
cmd:option('--finetune_lrMult', 1, 'if <0, no change in learning rate, else learning rate for pretrained modules is multiplied by this factor')
cmd:option('--finetuneRNNFlag', -1, 'in finetune mode, whether or not to slow aggregator training')
cmd:option('--finetuneDecoderFlag', -1, 'in finetune mode, whether or not to slow decoder training')
cmd:option('--finetuneActorFlag', -1, 'in finetune mode, whether or not to slow actor training')
cmd:option('--monitorParamShareFlag', -1, 'whether to monitor parameter sharing at each callback... only for debugging')
--cmd:option('--shareReconstructorsFlag', 1, 'whether or not to share reconstructors in greedy mode')
cmd:option('--actOnPixels', -1, 'whether or not actor module should see actual image pixels')
cmd:option('--actOnTime', 1, 'whether or not actor module should get time as input')
cmd:option('--actOnElev', 1, 'whether or not actor module should get absolute elevation as input')
cmd:option('--actOnAzim', -1, 'whether or not actor module should get absolute azimuth as input')
cmd:option('--knownElev', 1, 'whether or not location module should get absolute elevation as input')
cmd:option('--knownAzim', -1, 'whether or not location module should get absolute azimuth as input')
--cmd:option('--network', 'NA', 'network to be used for feature extraction - vgg_m | vgg_m_modelnetft10303 | vgg_m_modelnetft10303_early')
--cmd:option('--layer', 'NA', 'layer to be used for feature extraction - fc7 | conv5')
--cmd:option('--pixel_input', 1, 'whether or not to replace layer features with just pixel values')
cmd:option('--mean_subtract_input', -1, 'whether or not to replace layer features with just pixel values')
cmd:option('--mean_subtract_output', 1, 'whether or not to replace layer features with just pixel values')
cmd:option('--rotationCompensationFlag', 1, 'whether or not the system output is rotated before computing loss or system performance')
cmd:option('--compensateKnownPosFlag', -1, '(if rotationCompensationFlag), whether the output viewgrid is rotated even along known coordinate ...')
cmd:option('--patchSensorReLU', 1, 'set to -1 for backward compatibility with older versions where there was no ReLU at the end of patchSensor');
cmd:option('--inplaceReLU', 1, 'set to -1 for backward compatibility with older versions where there was no ReLU at the end of patchSensor');

--[[ reinforce ]]--
cmd:option('--rewardScale', 1, "scale of positive reward at the time rho")
cmd:option('--stepwiseRewardFactor', 0, "scale of positive reward for time 1 through rho-1 = stepwiseRewardFactor * rewardScale")
cmd:option('--discountFactor', 1, "when 1, all future rewards weighted equally. when 0, only current time-step reward accounted for.")
cmd:option('--rewardOffset', 0, "VRRegressReward will use this as offset...")

--[[ glimpse layer ]]--
cmd:option('--actionsType', 'actor', 'can be one of: actor | presetViews | presetActions | random');
cmd:option('--straightThroughFlag', -1, '(applies to actor module) if >1, then backward pass ignores sigmoid module');
cmd:option('--presetViewpoints', {torch.Tensor{3,2}, torch.Tensor{3,7}, torch.Tensor{0,5}, torch.Tensor{3,5}, torch.Tensor{6,5}}, 'if presetViewpointsFlag is true, actor module is never actually used. Instead the given sequence of viewpoints is traced.')
--cmd:option('--presetActionsFlag', 0, 'if true, actor module is never actually used. Instead the given sequence of viewpoints is traced.')
cmd:option('--presetActions', {torch.Tensor{0,0}, torch.Tensor{0,5}, torch.Tensor{-3,-2}, torch.Tensor{3,0}, torch.Tensor{3,0}, torch.Tensor{0,-3}}, 'if presetActionsFlag is true, actor module is never actually used. Instead the given sequence of actions is performed.')
cmd:option('--glimpseHiddenSize', 256, 'size of glimpse hidden layer')
--cmd:option('--glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
--cmd:option('--glimpseDepth', 1, 'number of concatenated downscaled patches')
cmd:option('--locatorHiddenSize', 16, 'size of actuator hidden layer')
cmd:option('--imageHiddenSize', 256, 'size of hidden layer combining glimpse and actor hiddens')

--[[ recurrent layer ]]--
cmd:option('--rho', 3, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--action_gridsize_factor', 1, '(only when action_gridshape is <0) factor p => action grid is approximately p times smaller than grid');
cmd:option('--action_transfer', 'SoftMax', 'how are action probabilities forced to be positive?');
cmd:option('--hiddenSize', 256, 'number of hidden units used in Simple RNN.')
cmd:option('--dropout', false, 'apply dropout on hidden neurons')

--[[ data ]]--
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation')
cmd:option('--trainOnly', false, 'forget the validation and test sets, focus on the training set')
cmd:option('--noTest', -1, 'if true, no test set')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--xpPath', '', 'path to a previously saved experiment, for the purpose of resuming')
cmd:option('--initModel', '', 'path to a previously saved model, so that the random initialization can be replaced by the saved model weights')
cmd:option('--replaceActor', -1, '(if initModel) if true, then will the actor of initModel with the newly initialized actor (useful esp. when initModel was for rho=1, and you want to try variations in actor architecture etc.)')
cmd:option('--resumeFromSnap', '', 'path to a previously saved experiment, for the purpose of loading the model and copying (some of) its weights')


cmd:option('--pretrainModeEpochs', 0, 'number of epochs to spend on pretraining lookahed module');
cmd:option('--action_gridshape', -1, 'limits the range of actions that may be selected');
cmd:option('--loggerfile', './outputs/loggerfile.rec', 'where optim.Logger stores results');
cmd:option('--jobno', 0, 'used when entering results into results table');
cmd:option('--greedyLossFlag', 1, 'whether or not to compute reconstruction penalty for each time step (if false, computed only at end)');
cmd:option('--monitorUpdateRatio', -1, 'whether or not to monitor layerwise update ratios');
cmd:option('--trackSelActions', 1, 'whether or not to display action selection histograms per epoch');

dp.SAVE_DIR='./outputs/models/';

--[[ debug ]]--
cmd:option('--dataset', 'SUN360', 'sun360 dataset')
cmd:option('--unseenCategoryTest', -1, 'if true, testing on modelnet-30 rather than on modelnet-10')
cmd:option('--unseenCategoryVal', -1, 'if true, validation on modelnet-30 rather than on modelnet-10')
cmd:option('--full_data', -1, 'false=>reduce size of dataset')
cmd:option('--num_trn_samples', -1, '(applied after full data option) -1=>do not curtail dataset further')  -- enables faster training when used
cmd:option('--no_loadInit', false, 'false=>load previous weights')
cmd:option('--manual_seed', -1, 'manual seed for torch, cutorch and xp')
cmd:option('--display_live', false, 'false=>store plots to files only')

cmd:option('--sys_cmd', '', 'system command to execute every $(sys_cmd_iter) epochs')
cmd:option('--sys_cmd_iter', 1, 'epoch gap at which to execute sys_cmd')
cmd:option('--sys_cmd2', '', 'system command to execute every $(sys_cmd2_iter) epochs')
cmd:option('--sys_cmd2_iter', 1, 'epoch gap for sys_cmd2')
cmd:option('--report_res_iter', 20, 'epoch gap at which to report results')

cmd:option('--evaluation', false, 'set this to not evaluate after the experiment (useful while debugging)')
cmd:option('--overwrite', false, 'set this to overwrite a previous model at this location')
cmd:option('--resultfileroot', 'tmp', 'name of file to print results to')

cmd:text()
opt = cmd:parse(arg or {})

opt.inplaceReLU = (opt.inplaceReLU>0) and true or false
opt.mean_subtract_input = (opt.mean_subtract_input>0) and true or false
opt.mean_subtract_output = (opt.mean_subtract_output>0) and true or false
opt.straightThroughFlag = (opt.straightThroughFlag>0) and true or false
opt.replaceActor = (opt.replaceActor>0) and true or false
opt.rotationCompensationFlag = (opt.rotationCompensationFlag>0) and true or false
opt.compensateKnownPosFlag = (opt.compensateKnownPosFlag>0) and true or false
opt.memorizeObservedViews = (opt.memorizeObservedViews>0) and true or false
opt.noTest = (opt.noTest>0) and true or false
opt.greedyLossFlag = (opt.greedyLossFlag>0) and true or false
opt.initFeedbackFlag = (opt.initFeedbackFlag>0) and true or false
opt.identFeedbackFlag = (opt.identFeedbackFlag>0) and true or false
opt.avgAtTestFlag = (opt.avgAtTestFlag>0) and true or false
opt.biasDecayFlag = (opt.biasDecayFlag>0) and true or false
opt.batchNormFlag = (opt.batchNormFlag>0) and true or false
opt.finetuneDecoderFlag = (opt.finetuneDecoderFlag>0) and true or false
opt.finetuneRNNFlag = (opt.finetuneRNNFlag>0) and true or false
opt.finetuneActorFlag = (opt.finetuneActorFlag>0) and true or false
opt.monitorParamShareFlag= (opt.monitorParamShareFlag>0) and true or false
opt.noMemory = (opt.noMemory>0) and true or false
opt.actOnPixels = (opt.actOnPixels>0) and true or false
opt.actOnTime = (opt.actOnTime>0) and true or false
opt.actOnElev = (opt.actOnElev>0) and true or false
opt.actOnAzim = (opt.actOnAzim>0) and true or false
opt.knownElev = (opt.knownElev>0) and true or false
opt.knownAzim = (opt.knownAzim>0) and true or false
opt.monitorUpdateRatio = (opt.monitorUpdateRatio>0) and true or false
opt.trackSelActions = (opt.trackSelActions>0) and true or false
opt.unseenCategoryTest = (opt.unseenCategoryTest>0) and true or false
opt.unseenCategoryVal = (opt.unseenCategoryVal>0) and true or false
opt.wrapAroundRows = (opt.wrapAroundRows>0) and true or false
opt.wrapAroundCols = (opt.wrapAroundCols>0) and true or false
--opt.cyclicalLRFlag = (opt.cyclicalLRFlag>0) and true or false
opt.full_data = (opt.full_data>0) and true or false

opt.origLR=opt.learningRate;
opt.finalLR = opt.finalLR_ratio*opt.origLR;
opt.rewardScale = (opt.randomActionsFlag or opt.presetViewpointsFlag or opt.presetActionsFlag) and 0 or opt.rewardScale
opt.stepwiseRewardScale = opt.stepwiseRewardFactor*opt.rewardScale
opt.actOnElev = opt.knownElev and opt.actOnElev or false -- can only act on elev when elev is known
opt.actOnAzim = opt.knownAzim and opt.actOnAzim or false

opt.step_exp_decayFactor = opt.step_decayFactor -- for backward compatibility

if opt.actionsType == "actor" then
  -- dummy
elseif opt.actionsType == "random" then
  opt.randomActionsFlag=true;
elseif opt.actionsType == "presetViews" then
  opt.presetViewpointsFlag=true;
elseif opt.actionsType == "presetActions" then
  opt.presetActionsFlag=true;
else
  abort()
end

if not opt.no_loadInit then
--if 1 then
  torch.manualSeed(1151);
  if opt.cuda then
    require 'cunn';
    cutorch.manualSeed(1151);
  end
  xp_seed=1151;
elseif opt.manual_seed>0 then
  torch.manualSeed(opt.manual_seed);
  if opt.cuda then
    require 'cunn'
    cutorch.manualSeed(opt.manual_seed);
  end
  xp_seed=opt.manual_seed;
end

if opt.jobno~=0 and not opt.overwrite then
  -- check whether stored model already exists
  local model_filename = dp.SAVE_DIR .. opt.jobno .. '.dat'
  assert(lfs.attributes(model_filename)==nil, 'Previous model ' .. model_filename .. ' found. Quitting.')
end

--require('mobdebug').start()
if not opt.silent then
  table.print(opt)
end

if opt.xpPath ~= '' then
  -- check that saved model exists
  assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
end

require 'sun360'
ds = dp.SUN360{
  --network = opt.network,
  --layer = opt.layer,
  --pixel_input = opt.pixel_input,
  mean_subtract_input = opt.mean_subtract_input,
  mean_subtract_output = opt.mean_subtract_output,
  mini = not opt.full_data,
  unseenCategoryTest = opt.unseenCategoryTest,
  unseenCategoryVal = opt.unseenCategoryVal,
  noTest = opt.noTest
}


local num_trn_samples=ds:trainSet():inputs()._input:size(1);
local num_val_samples=ds:validSet():inputs()._input:size(1);
local num_tst_samples=nil
if not opt.noTest then
  num_tst_samples=ds:testSet():inputs()._input:size(1);
end
local iters_per_epoch=torch.ceil(num_trn_samples/opt.batchSize)
if not opt.noTest then
  print("Num_samples: (" .. num_trn_samples .. ", " .. num_val_samples  .. ", " .. num_tst_samples .. ")");
else
  print("Num_samples: (" .. num_trn_samples .. ", " .. num_val_samples  .. ")");
end

---- each channel must encode one target
--tmp=ds:trainSet():targets()._input:clone();
--num_samples=tmp:size(1);
--tmp=tmp:repeatTensor(2*opt.rho+1,1,1); -- repeat along the channels dimension...
----tmp:narrow(1,opt.rho+2,opt.rho):zero(); -- setting last rho channels to zero in preparation for the lookahead loss..
--ds:trainSet():targets()._input=tmp;
--ds:trainSet():targets()._view='cbw';
--
--tmp=ds:validSet():targets()._input:clone();
--num_samples=tmp:size(1);
--tmp=tmp:repeatTensor(2*opt.rho+1,1,1); -- repeat along the channels dimension...
----tmp:narrow(1,opt.rho+2,opt.rho):zero(); -- setting last rho channels to zero in preparation for the lookahead loss..
--ds:validSet():targets()._input=tmp;
--ds:validSet():targets()._view='cbw';
--
--if not opt.noTest then
--  tmp=ds:testSet():targets()._input:clone();
--  num_samples=tmp:size(1);
--  tmp=tmp:repeatTensor(2*opt.rho+1,1,1); -- repeat along the channels dimension...
--  --tmp:narrow(1,opt.rho+2,opt.rho):zero(); -- setting last rho channels to zero in preparation for the lookahead loss..
--  ds:testSet():targets()._input=tmp;
--  ds:testSet():targets()._view='cbw';
--end

--[[Saved experiment]]--
if opt.xpPath ~= '' then
  if opt.cuda then
    require 'cunn'
    if opt.useDevice>-1 then
      cutorch.setDevice(opt.useDevice)
    end
  end
  xp = torch.load(opt.xpPath)
  if opt.cuda then
    xp:cuda()
  else
    xp:float()
  end
  xp:run(ds)
  os.exit()
end

print("Defining model");
local num_channels = 3;
--[[Model]]--
if opt.action_gridshape>0 then
  opt.action_gridshape = {opt.action_gridshape, opt.action_gridshape};
else
  opt.action_gridshape = torch.totable(torch.ceil(ds.gridshape:float()*opt.action_gridsize_factor):int())
end
if opt.action_gridshape[1]%2 == 0 then
  opt.action_gridshape[1]=opt.action_gridshape[1]+1;
end
if opt.action_gridshape[2]%2 == 0 then
  opt.action_gridshape[2]=opt.action_gridshape[2]+1;
end
assert(opt.action_gridshape[1]%2==1)
assert(opt.action_gridshape[2]%2==1)
local num_actions= opt.action_gridshape[1]*opt.action_gridshape[2];
-- define action values
opt.action_values={{},{}};
for j=1,2 do
  for i=1,opt.action_gridshape[j] do
    opt.action_values[j][i]=i-(opt.action_gridshape[j]+1)/2;
  end
  opt.action_values[j]= torch.DoubleTensor(opt.action_values[j]);
  print(opt.action_values[j]);
end
-- actors
assert(opt.explorationBaseFactor>=0);
assert(opt.explorationDecay>=0 and opt.explorationDecay<=1);
local actor=nn.Identity() -- just initializing to something for the case when actionsType is not actor
if opt.actionsType=="actor" then
  local act_ipsz=opt.hiddenSize
  act_ipsz=act_ipsz+2 -- for relative camera positions
  act_ipsz=act_ipsz+ (opt.actOnTime and 1 or 0)
  act_ipsz=act_ipsz+ (opt.actOnElev and 1 or 0) -- for true camera position
  act_ipsz=act_ipsz+ (opt.actOnAzim and 1 or 0)
  local aggregAct_mod = nn.Sequential() -- to act on outputs of aggregator
  aggregAct_mod:add(nn.Linear(act_ipsz, 128))
  aggregAct_mod:add(nn.ReLU(opt.inplaceReLU));
  aggregAct_mod:add(nn.Linear(128, 128))
  aggregAct_mod:add(nn.ReLU(opt.inplaceReLU));
--aggregAct_mod = nn.DummyContainer(aggregAct_mod)
  --aggregAct_mod:add(nn.Linear(128, num_actions))

  local pixelAct_mod = nn.Sequential()
  local combineAct_mod = nn.Sequential()
  if opt.actOnPixels then
    pixelAct_mod:add(nn.SpatialConvolution(num_channels,32,5,5,1,1,2,2)); -- conv1
    pixelAct_mod:add(nn.SpatialMaxPooling(3,3,2,2)); -- pool1
    pixelAct_mod:add(nn.ReLU(opt.inplaceReLU)); -- relu1
    pixelAct_mod:add(nn.SpatialConvolution(32,32,5,5,1,1,2,2)); --conv2
    pixelAct_mod:add(nn.ReLU(opt.inplaceReLU)); --relu2
    pixelAct_mod:add(nn.SpatialAveragePooling(3,3,2,2)); --pool2
    pixelAct_mod:add(nn.SpatialConvolution(32,64,5,5,1,1,2,2)); -- conv3
    pixelAct_mod:add(nn.ReLU(opt.inplaceReLU)); --relu3
    pixelAct_mod:add(nn.SpatialAveragePooling(3,3,2,2)); --pool3 -- output should be 64 maps of size 3x3
    pixelAct_mod:add(nn.Collapse(3)); --output is of dimension 576
    pixelAct_mod:add(nn.Linear(576,128))

    combineAct_mod:add(nn.JoinTable(1,1))
    combineAct_mod:add(nn.Linear(256, 128)) -- ip2 (256)
    combineAct_mod:add(nn.ReLU(opt.inplaceReLU))
    --combineAct_mod:add(nn.Linear(128, num_actions)) --ip3
    --combineAct_mod:add(nn.BatchNormalization(num_actions))
  end

  local presample_mod
  if opt.actOnPixels then
    presample_mod = nn.Sequential();
    presample_mod:add(
      nn.ParallelTable():add(
        aggregAct_mod):add(
        pixelAct_mod)
      ):add(combineAct_mod)
   else
     presample_mod = aggregAct_mod
   end
   -- presample_mod:add(nn.Linear(128,num_actions)):add(nn.BatchNormalization(num_actions)) (used earlier, but forces selection of bad actions?)
   presample_mod:add(nn.BatchNormalization(128)):add(nn.Linear(128,num_actions))

  local sigmoid_mod = nn.Sequential()
  if not opt.action_transfer then
    sigmoid_mod:add(nn.HardTanh(0,1)) -- bounds probs between 0 and 1
  else
    sigmoid_mod = nn.Sequential();
    if opt.action_transfer=="SoftMax" then
      sigmoid_mod:add(nn.MulConstant(1/(opt.explorationBaseFactor+1))); -- temperature = 1+explorationBaseFactor, no effect when explorationBaseFactor=0
    end
    sigmoid_mod:add(nn[opt.action_transfer]()) -- Abs, or ReLU, for instance
  end
  sigmoid_mod:add(nn.Normalize(1)) -- each row sums to 1
  if not opt.action_transfer == "SoftMax" then
    sigmoid_mod:add(nn.AddConstant(opt.explorationBaseFactor/num_actions)); -- ensuring all actions have at least approximately 1/num_actions*10 probability (for exploration at training time)
    sigmoid_mod:add(nn.Normalize(1)) -- each row sums to 1
  end
  local stochastic_eval = false -- TODO make this a parameter

  local reinforce_mod = nn.ReinforceCategorical(stochastic_eval);
  local postsample_mod = nn.ArgMax(2)

  actor = nn.ActorMod{
    presample_mod = presample_mod,
    sigmoid_mod = sigmoid_mod,
    reinforce_mod = reinforce_mod,
    postsample_mod = postsample_mod,
    straightThroughFlag = opt.straightThroughFlag
  }
end
-- glimpse network (rnn input layer)
local locationSensor = nn.Sequential()
local location_ipsz=2; -- relative camera position
location_ipsz=location_ipsz+1; -- time
location_ipsz=location_ipsz+(opt.knownElev and 1 or 0)
location_ipsz=location_ipsz+(opt.knownAzim and 1 or 0)
locationSensor:add(nn.Linear(location_ipsz, opt.locatorHiddenSize))
locationSensor:add(nn.ReLU(opt.inplaceReLU))

local patchSensor

patchSensor=nn.Sequential()
patchSensor:add(nn.Dropout(opt.featDropout));
--patchSensor:add(nn.Reshape(1,32,32)); -- C*W*H, because reshape operates row-wise in Torch, and column-wise in Matlab
patchSensor:add(nn.SpatialConvolution(num_channels,32,5,5,1,1,2,2)); -- conv1
patchSensor:add(nn.SpatialMaxPooling(3,3,2,2)); -- pool1
patchSensor:add(nn.ReLU(opt.inplaceReLU)); -- relu1
patchSensor:add(nn.SpatialConvolution(32,32,5,5,1,1,2,2)); --conv2
patchSensor:add(nn.ReLU(opt.inplaceReLU)); --relu2
patchSensor:add(nn.SpatialAveragePooling(3,3,2,2)); --pool2
patchSensor:add(nn.SpatialConvolution(32,64,5,5,1,1,2,2)); -- conv3
patchSensor:add(nn.ReLU(opt.inplaceReLU)); --relu3
patchSensor:add(nn.SpatialAveragePooling(3,3,2,2)); --pool3 -- output should be 64 maps of size 3x3
patchSensor:add(nn.Collapse(3)); --output is of dimension 576
patchSensor:add(nn.Linear(576,opt.glimpseHiddenSize)) -- ip1, output is of dimension opt.glimpseHiddenSize=256
if opt.patchSensorReLU then
  patchSensor:add(nn.ReLU(opt.inplaceReLU))
end

local combineSensors=nn.Sequential()
combineSensors:add(nn.JoinTable(1,1))

combineSensors:add(nn.Linear(opt.glimpseHiddenSize+opt.locatorHiddenSize, opt.imageHiddenSize)) -- ip2 (256)
combineSensors:add(nn.ReLU(opt.inplaceReLU))
combineSensors:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize)) --ip3

if opt.batchNormFlag then
  combineSensors:add(nn.BatchNormalization(opt.hiddenSize))
end
combineSensors:add(nn.Dropout(opt.combineDropout))

local aggregator
if opt.aggregatorStyle == "rnn" then
  local startModule = nn.Identity()
  local inputModule = nn.Identity();
  local rec_feedback;
  if opt.noMemory then
    rec_feedback = nn.MulConstant(0)
  else
    rec_feedback = nn.Sequential():add(nn.Linear(opt.hiddenSize, opt.hiddenSize)):add(nn.ReLU())
  end
  local transferModule = nn.Sequential():add(nn.Normalize(1)):add(nn.Add(opt.hiddenSize)):add(nn.ReLU()); -- something scale invariant here so that output is scale invariant
  local rho = 9999
  local mergeModule = nn.CAddTable();
  aggregator = nn.Recurrent(startModule, inputModule, rec_feedback, transferModule, rho, mergeModule)
elseif opt.aggregatorStyle == "lstm" then
  local rho = 9999
  local cell2gate = true
  local dropout_p = 0
  local mono = false -- not sure what this is, but it has only to do  with dropout, so needn't worry about this as long as dropout_p is zero
  aggregator = nn.LSTM(opt.hiddenSize, opt.hiddenSize, rho, cell2gate, dropout_p, mono);
  if opt.noMemory then
    abort()
  end
elseif opt.aggregatorStyle == "fastlstm" then
  local startModule = nn.Identity()
  local inputModule = nn.Identity();
  local rec_feedback;
  if opt.noMemory then
    rec_feedback = nn.MulConstant(0)
  else
    rec_feedback = nn.Sequential():add(nn.FastLSTM(opt.hiddenSize, opt.hiddenSize)):add(nn.ReLU())
  end
  local transferModule = nn.Sequential():add(nn.Normalize(1)):add(nn.Add(opt.hiddenSize)):add(nn.ReLU()); -- something scale invariant here so that output is scale invariant
  local rho = 9999
  local mergeModule = nn.CAddTable();
  aggregator = nn.Recurrent(startModule, inputModule, rec_feedback, transferModule, rho, mergeModule)
else
  abort()
end

imageSize = ds:imageSize('h')
-- reconstructor:
local reconstructor = nn.Sequential()
reconstructor:add(nn.Normalize(1));
if opt.batchNormFlag then
  -- TODO CHANGE IF USING CONVOLUTIONAL FEATURE MAP, NOT FC FEATURES
  reconstructor:add(nn.BatchNormalization(opt.hiddenSize));
end
reconstructor:add(nn.Linear(opt.hiddenSize,1024)):add(nn.LeakyReLU(0.2,true)):add(nn.Reshape(64,4,4)); -- output Nx64x4x4
reconstructor:add(nn.SpatialFullConvolution(64,256,5,5,2,2,2,2,1,1)):add(nn.LeakyReLU(0.2,true)); --output Nx256x8x8
reconstructor:add(nn.SpatialFullConvolution(256,128,5,5,2,2,2,2,1,1)):add(nn.LeakyReLU(0.2,true)); --output Nx128x16x16
reconstructor:add(nn.SpatialFullConvolution(128,num_channels*ds.gridshape[1]*ds.gridshape[2],5,5,2,2,2,2,1,1)); --output Nx84x32x32 (84 is the number of views I want to produce. For a standard auto-encoder this should be 1)
if not opt.mean_subtract_output then
  reconstructor:add(nn.ReLU(opt.inplaceReLU)); -- can force to be fully positive
else
  reconstructor:add(nn.LeakyReLU(0.8, true)); -- must not be forced to be positive
end
reconstructor:add(nn.View(ds.gridshape[1]*num_channels,ds.view_snapshape[2]*ds.gridshape[2],ds.view_snapshape[1]):setNumInputDims(3))
reconstructor:add(nn.Transpose({3,4}))
reconstructor:add(nn.View(num_channels,ds.view_snapshape[1]*ds.gridshape[1],ds.view_snapshape[2]*ds.gridshape[2]):setNumInputDims(3)) -- montage
reconstructor:add(nn.Contiguous())
--reconstructor:add(nn.SpatialConvolution(3,3,5,5,1,1,2,2)) -- just to stitch the montage smoothly?

local rotator = nn.CircShift();  -- compensate for the rotation ...

local vectorize_reconstruction = nn.Sequential():add( -- vector produced in the same way as target vector
    nn.Transpose({3,4})):add(
    nn.View(num_channels*ds.view_snapshape[1]*ds.view_snapshape[2]*ds.gridshape[1]*ds.gridshape[2]):setNumInputDims(3)
    )

local attention = nn.ActiveMod{
  finetuneFactor = opt.finetune_lrMult,
  action_lr_mult = opt.finetuneActorFlag and opt.finetune_lrMult or 1,
  rnn_lr_mult = opt.finetuneRNNFlag and opt.finetune_lrMult or 1,
  decoder_lr_mult = opt.finetuneDecoderFlag and opt.finetune_lrMult or 1,
  invarianceFlag=false,
  randomActionsFlag=opt.randomActionsFlag,
  presetViewpoints = opt.presetViewpointsFlag and opt.presetViewpoints or {},
  presetActions = opt.presetActionsFlag and opt.presetActions or {},
  rnnMod=aggregator,
  locationMod=locationSensor,
  patchMod=patchSensor,
  combineMod=combineSensors,
  actionMod=actor,
  reconstructMod=reconstructor,
  rotateMod = rotator,
  vectorizeMod = vectorize_reconstruction,
  nStep=opt.rho,
  actOnPixels=opt.actOnPixels,
  view_gridshape=torch.totable(ds.gridshape),
  action_gridshape=opt.action_gridshape,
  glimpseshape=torch.totable(ds.feat_snapshape),
  viewshape=torch.totable(ds.view_snapshape),
  pretrainModeIters=opt.pretrainModeEpochs*iters_per_epoch,
  startAnywhereFlag=true, -- so that each "grid" may be started at any position, randomly... effectively data augmentation.
  action_values={opt.action_values[1]:clone(), opt.action_values[2]:clone()}, -- specifies the meaning of each action value
  wrapAroundFlag = {opt.wrapAroundCols, opt.wrapAroundRows}, -- if false, then rows and cols get "capped" at the boundaries, rather than wrapping around the other side.
  avgAtTestFlag = opt.avgAtTestFlag,
  zeroFirstStepFlag = true, -- so that first view is truly random (this is best for GERMS data quirks, but maybe good for all data)
  actOnTime = opt.actOnTime,
  actOnElev = opt.actOnElev,
  actOnAzim = opt.actOnAzim,
  knownAzim = opt.knownAzim,
  knownElev = opt.knownElev,
  rotationCompensationFlag = opt.rotationCompensationFlag,
  compensateKnownPosFlag = opt.compensateKnownPosFlag,
  memorizeObservedViews = opt.memorizeObservedViews,
  mean_subtract_input = opt.mean_subtract_input,
  mean_subtract_output = opt.mean_subtract_output,
  average_viewgrid = ds._avg_viewgrid:transpose(2,3),
}


-- model is a reinforcement learning agent
agent = nn.Sequential()
agent:add(nn.Convert(ds:ioShapes(), 'bchw'))
agent:add(attention)

top=nn.ConcatTable();
for step=1,opt.rho do
  top:add(nn.SelectTable(step));
end

-- step-wise reinforcement reward evaluation pipeline
local reinforce_input=nn.ConcatTable()
reinforce_input:add(nn.Identity()) -- output is the reconstruction table
local baseline_input=nn.ConcatTable() -- output is the table of baselines for each step
local seq = {}
for step =1,opt.rho do
  seq[step]=nn.Sequential()
  seq[step]:add(nn.SelectTable(-1))
  seq[step]:add(nn.Constant(0,1))
  if step==opt.rho or opt.stepwiseRewardScale>0 then -- to avoid changing the number of parameters, and therefore the random seed-related setting
    seq[step]:add(nn.Add(1))
  end
  baseline_input:add(seq[step])
end
reinforce_input:add(baseline_input) -- output is {table1 , table2}
-- table1 = {recon_1, recon_2, ..., recon_rho}
-- table2 = {base_1, base_2, ..., base_rho}

top:add(reinforce_input)
-- output is rho+1-element table: {recon1, recon2, ..., recon_rho, {{recon1, recon2, ... recon_rho}, {base1, base2, ..., base_rho}}}
recon_final=opt.rho;
agent:add(top)

if opt.initStyle == "xavier" then
  require('../misc/torch-toolbox/Weight-init/weight-init.lua')(agent, 'xavier');
elseif opt.initStyle == "uniform" then
  for k,param in ipairs(agent:parameters()) do
    param:uniform(-opt.initLim, opt.initLim)
  end
elseif opt.initStyle == "uniform_but_constantActor" then
  for k,param in ipairs(agent:parameters()) do
    param:uniform(-opt.initLim, opt.initLim)
  end
  for k,param in ipairs(actor:parameters()) do
    param:zero():add(opt.initLim)
  end
else
  -- TODO also implement xavier bt constantActor?
  abort()
end
for step=1, opt.rho do
  local tmp=seq[step]:findModules('nn.Add')
  tmp = tmp and tmp[1] -- check if found
  if tmp then
    tmp.bias[1]=0; --setting baseline to be zero at the beginning
  end
end


function table.containsKey(table, element)
  for key,_ in pairs(table) do
    if key == element then
      return true
    end
  end
  return false
end

function table.containsValue(table, element)
  for _,value in pairs(table) do
    if value == element then
      return true
    end
  end
  return false
end

function size2str(sz)
  local str= "";
  for i=1,#sz do
    str= str .. (i>1 and "x" or "") .. sz[i];
  end
  return str
end

if opt.initModel~='' and opt.initModel~='NA' then
  local orig_agent=torch.load(opt.initModel);
  local orig_agent = orig_agent.module or orig_agent.modules[1];
  local orig_ra = orig_agent:findModules('nn.ActiveMod')[1];
  print('Initializing with ' .. opt.initModel);
  agent:findModules('nn.ActiveMod')[1]:__init{ -- copying parameters from orig_ra
    finetuneFactor = opt.finetune_lrMult,
    action_lr_mult = opt.finetuneActorFlag and opt.finetune_lrMult or 1,
    rnn_lr_mult = opt.finetuneRNNFlag and opt.finetune_lrMult or 1,
    decoder_lr_mult = opt.finetuneDecoderFlag and opt.finetune_lrMult or 1,
    invarianceFlag=false,
    randomActionsFlag=opt.randomActionsFlag,
    presetViewpoints = opt.presetViewpointsFlag and opt.presetViewpoints or {},
    presetActions = opt.presetActionsFlag and opt.presetActions or {},
    rnnMod     = orig_ra.rnn,
    locationMod= orig_ra.location,
    patchMod   = orig_ra.patch,
    combineMod = orig_ra.combine,
    actionMod  = opt.replaceActor and
                 actor or
                 orig_ra.action,
    reconstructMod= orig_ra.reconstruct,
    rotateMod =  orig_ra.rotation,
    vectorizeMod = orig_ra.vectorize,
    nStep=opt.rho,
    actOnPixels = opt.replaceActor and
                      opt.actOnPixels or
                      (orig_ra.actOnPixels and orig_ra.actOnPixels or false), -- if orig_ra had no actOnPixels, sets to false by default
    view_gridshape=torch.totable(ds.gridshape),
    action_gridshape=opt.action_gridshape,
    glimpseshape=torch.totable(ds.feat_snapshape),
    viewshape=torch.totable(ds.view_snapshape),
    pretrainModeIters=opt.pretrainModeEpochs*iters_per_epoch,
    startAnywhereFlag=true, -- so that each "grid" may be started at any position, randomly... effectively data augmentation.
    action_values={opt.action_values[1]:clone(), opt.action_values[2]:clone()}, -- specifies the meaning of each action value
    wrapAroundFlag = {opt.wrapAroundCols, opt.wrapAroundRows}, -- if false, then rows and cols get "capped" at the boundaries, rather than wrapping around the other side.
    avgAtTestFlag = opt.avgAtTestFlag,
    zeroFirstStepFlag = true, -- so that first view is truly random (this is best for GERMS data quirks, but maybe good for all data)
    actOnTime = opt.actOnTime,
    actOnElev = opt.actOnElev,
    actOnAzim = opt.actOnAzim,
    knownAzim = opt.knownAzim,
    knownElev = opt.knownElev,
    rotationCompensationFlag = opt.rotationCompensationFlag,
    compensateKnownPosFlag = opt.compensateKnownPosFlag,
    memorizeObservedViews = opt.memorizeObservedViews,
    mean_subtract_input = opt.mean_subtract_input,
    mean_subtract_output = opt.mean_subtract_output,
    average_viewgrid = ds._avg_viewgrid:transpose(2,3),
  }
  agent:double()
  --agent:float()
  print("Re-initialized ActiveMod module with copied weights!")
  print("WARNING: NOT COPYING BASELINE REWARD (TODO for some cases)")
  --ra=agent:findModules('nn.ActiveMod')[1];
end

ra=agent:findModules('nn.ActiveMod')[1];
model_params, model_gradParams=agent:getParameters();
actor_params, actor_gradParams = ra['action']:parameters()

if (not opt.noMemory) and (opt.initFeedbackFlag or opt.identFeedbackFlag) and opt.aggregatorStyle=="rnn" then
  -- initialize recurrent feedback weights to identity, to mimic averaging(+noise)
  local lin_feedback = ra.rnn.feedbackModule:findModules('nn.Linear')[1]
  w_size=lin_feedback.weight:size(1);
  assert(w_size == lin_feedback.weight:size(2));
  if opt.identFeedbackFlag then
    lin_feedback.weight=torch.eye(w_size, w_size);
    lin_feedback.bias:zero();
  elseif opt.initFeedbackFlag and opt.initFeedbackType=="identity" then
  -- noisy version of identity
    lin_feedback.weight=torch.eye(w_size, w_size);
    lin_feedback.bias:zero();
    if false then
      local var=(opt.uniform>0 and opt.uniform or 0.1)*0.1;
      lin_feedback.weight:copy(torch.rand(w_size,w_size)*var);
      for i=1,w_size do
        lin_feedback.weight[i][i]=(torch.rand(1)-0.5)*var+1;
      end
      lin_feedback.bias:copy(torch.rand(w_size)*0.01); --very small biases
    end
  elseif opt.initFeedbackFlag and opt.initFeedbackType=="zero" then
  local var=(opt.uniform>0 and opt.uniform or 0.1)*0; --very small weights
    lin_feedback.weight:copy(torch.rand(w_size,w_size)*var);
    lin_feedback.bias:copy(torch.rand(w_size)*var); --very small biases
  else
  abort()
  end
end

if opt.resumeFromSnap~='' and opt.resumeFromSnap~='NA' then
  agent=torch.load(opt.resumeFromSnap);
  agent = agent.module or agent.modules[1];
  print('Resuming from ' .. opt.resumeFromSnap);
  ra=agent:findModules('nn.ActiveMod')[1];
  model_params, model_gradParams=agent:getParameters();
  actor_params, actor_gradParams = ra['action']:parameters()
end

print("# params");
print(#model_params);


--[[Propagators]]--
opt.decayRate = (opt.finalLR - opt.origLR)/opt.saturateEpoch
opt.decayFactor = math.exp(math.log(opt.finalLR/opt.origLR)/opt.saturateEpoch);

local repeatingTarget = true
local learning_criterion=nn.ParallelCriterion(repeatingTarget)
for step=1, opt.rho do
  local weight= (step<opt.rho and not opt.greedyLossFlag) and 0 or 1 -- If not greedy, the last time step alone is weighted. Else, all time step classification losses are weighted.
  learning_criterion:add(nn.ModuleCriterion(nn[opt.reconstructionCriterion](), nil, nn.Convert()), weight) -- BACKPROP RECONSTRUCTION LOSS FOR EVERY TIME STEP. If not greedy, the last time step alone is weighted
end

learning_criterion:add(nn.ModuleCriterion(
  nn.VRRegressReward{
    module = agent,
    perstep_scale = opt.stepwiseRewardScale,
    finstep_scale = opt.rewardScale,
    offset = opt.rewardOffset,
    criterion = nn.MSECriterion(),
    sizeAverage = true,
    discount = opt.discountFactor,
    baseline_lr_factor = 15/opt.learningRate, -- so that no matter, what opt.learningRate, effective learning rate for baseline is always 15
  }, nil, nn.Convert()),
  1) -- REINFORCE reward

print("Defining loggers");
-- optim.Logger definition
local path=paths.dirname(opt.loggerfile);
local fname_ext=paths.basename(opt.loggerfile);
local ext=paths.extname(opt.loggerfile);

local fname=fname_ext:sub(1,fname_ext:find('[.]')-1);

local opt_fname= path .. '/' .. fname .. '.opts'

local mse_logger_name = path .. '/' .. fname .. '-reconMSE.' .. ext
mse_logger=optim.Logger(mse_logger_name)
local names={}
local styles={}
datasets = {'trn'}
if not opt.trainOnly then
  datasets[#datasets+1]='val'
  if not opt.noTest then
    datasets[#datasets+1]='tst'
  end
end
for _, dataset in ipairs(datasets) do
  for t = 1,opt.rho-1 do
    names[#names+1]= dataset .. '\\_mse\\_step' .. t
    styles[#styles+1]= '+'
  end
  names[#names+1]= dataset .. '\\_mse'
  styles[#styles+1]= '+-'
end
mse_logger:setNames(names);
mse_logger:style(styles);
mse_logger:setlogscale(true);

if not opt.display_live then
  mse_logger.showPlot=false
end

-- dp.Optimizer definition
local best_val_mse=1e10
local tst_at_best_val_mse=1e10
local best_val_epoch=0
local last_val_mse, last_tst_mse, last_trn_mse
local reconMSE_time=recon_final;

function deepcopy(orig)
   local orig_type = type(orig)
   local copy
   if orig_type == 'table' then
     copy = {}
     for orig_key, orig_value in next, orig, nil do
       copy[deepcopy(orig_key)] = deepcopy(orig_value)
     end
     setmetatable(copy, deepcopy(getmetatable(orig)))
   else -- number, string, boolean, etc
     copy = orig
   end
   return copy
end

function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

local selActionsHist={}
if opt.trackSelActions then
  for t=1,opt.rho do
  selActionsHist[t]=torch.zeros(num_actions);
  end
end

function save_best_res_snap()
  print "Saving best result so far"
  -- Saving params and results
  local tmp=1
  while lfs.attributes(opt.resultfileroot .. '.dat.lock')~=nil and tmp<1000 do
    sleep(0.1)
    -- print "Waiting to save"
    tmp=tmp+1
  end
  assert(tmp<1e5, 'could not save. file indefinitely locked.');
  os.execute('touch ' .. opt.resultfileroot .. '.dat.lock');
  if lfs.attributes(opt.resultfileroot .. '.dat')~=nil  then
    reports=torch.load(opt.resultfileroot .. '.dat')
  else
    reports={}
  end
  result={best_epoch=best_val_epoch, val_mse=best_val_mse, test_mse=tst_at_best_val_mse, trn_mse=trn_at_best_val_mse, converged=converged}
  print(opt.jobno)
  print(result)
  reports[opt.jobno]={params=nn.utils.recursiveType(deepcopy(opt),'torch.FloatTensor'), result=result};
  torch.save(opt.resultfileroot .. '.dat', reports);
  os.execute('rm ' .. opt.resultfileroot .. '.dat.lock');
end

print("Defining training, validation, testing functions")
if opt.lr_style=="step_if_stagnant" then
  best_trn_loss = torch.ones(opt.maxEpoch)*1e20;
  last_step_epoch=1;
end

function feedbackTable()
  local feedbackTable = {}
  feedbackTable[#feedbackTable+1] = dp.MSEFeedback{name = 'MSE_finstep', target_dim=1, output_module=nn.SelectTable(opt.rho)}
  for t = 1,opt.rho-1 do
    feedbackTable[#feedbackTable+1] = dp.MSEFeedback{name = 'MSE_step' .. t, target_dim=1, output_module=nn.SelectTable(t)}
  end
  return feedbackTable
end

train = dp.Optimizer{
  loss = learning_criterion,
  epoch_callback = function(model, report) -- called every epoch
    print(os.date("%X", os.time()) .. string.format(" ----- elapsed time: %.2fs\n", os.clock() - x))
    if report.epoch > 0 then
      if opt.actionsType == "actor" then
        if opt.trackSelActions then
          print("Selected actions histograms (last epoch): ");
          for t=2,opt.rho do
            print("Time " .. t .. ":");
            print(selActionsHist[t]:reshape(opt.action_gridshape[1], opt.action_gridshape[2]));
            selActionsHist[t]:zero(); -- resetting after every epoch
          end
        end
      end
      if opt.lr_style == 'constant' then
        -- dummy
      elseif opt.lr_style == 'linear' then
        opt.learningRate = opt.learningRate + opt.decayRate
        opt.learningRate = (opt.finalLR<opt.origLR) and math.max(opt.finalLR, opt.learningRate) or  math.min(opt.finalLR, opt.learningRate)
      --if not opt.silent then
      print("learningRate", opt.learningRate)
      --end
      elseif opt.lr_style=='step' then
        if report.epoch==opt.saturateEpoch then
          opt.learningRate=opt.finalLR;
        print("learningRate", opt.learningRate)
        end
      elseif opt.lr_style=='step_repeat' then
        if report.epoch%opt.saturateEpoch == 0 then
          opt.learningRate=opt.step_decayFactor*opt.learningRate;
          print("learningRate", opt.learningRate)
        end
      elseif opt.lr_style=='step_if_stagnant' then
        if report.epoch == opt.saturateEpoch then
          opt.learningRate=opt.finalLR;
          print("First learning rate step.");
          print("learningRate", opt.learningRate)
          last_step_epoch = report.epoch;
        end
        if last_step_epoch>1 and report.epoch > last_step_epoch + opt.stagnantWindow then -- shouldn't step immediately after previous step
          if best_trn_loss[report.epoch-opt.stagnantWindow-1] - best_trn_loss[report.epoch-1] < opt.stagnantDelta then
            print("Stagnant condition check triggered.")
            opt.learningRate=opt.step_decayFactor*opt.learningRate;
            print("learningRate", opt.learningRate)
            last_step_epoch = report.epoch
          end
        end
      elseif opt.lr_style=='step_exp'then
        if report.epoch==opt.saturateEpoch then
          opt.learningRate=opt.finalLR;
        print("learningRate", opt.learningRate)
        elseif report.epoch>opt.saturateEpoch then
        opt.learningRate=opt.learningRate*opt.step_decayFactor;
        print("learningRate", opt.learningRate)
        end
      elseif opt.lr_style=='exp' then
        if report.epoch<opt.saturateEpoch then
          opt.learningRate=opt.learningRate*opt.decayFactor;
        print("learningRate", opt.learningRate)
        end
      elseif opt.lr_style=='cyclical' then
        local idx=report.epoch%opt.saturateEpoch;
        local cycle_no = math.ceil(report.epoch/opt.saturateEpoch)
        if cycle_no%2==0 then
          -- even cycle => learning rate going from opt.finalLR to opt.origLR
          opt.learningRate=opt.finalLR+idx/opt.saturateEpoch*(opt.origLR-opt.finalLR)
        else
          -- odd cycle => learning rate going from opt.origLR to opt.finalLR
          opt.learningRate=opt.finalLR+idx/opt.saturateEpoch*(opt.finalLR-opt.origLR)
        end
        abort();
        print("learningRate", opt.learningRate)
      end

      if opt.actionsType=="actor" then
        if opt.explorationBaseFactor>0 and opt.explorationDecay > 0 then
          local clones = agent:findModules('nn.ActiveMod')[1].action.sharedClones
          for i=1,#clones do
            if opt.action_transfer == "SoftMax" then  -- for SoftMax, (1+explorationBaseFactor) behaves as temperature
              local tmp=clones[i]:findModules('nn.MulConstant')[1];
              local explorationFactor = (1/tmp.constant_scalar - 1)*(1-opt.explorationDecay); -- temperature decays to 1.0
              tmp.constant_scalar = 1/(explorationFactor+1);
            else
              local tmp=clones[i]:findModules('nn.AddConstant')[1];
              tmp.constant_scalar = tmp.constant_scalar*(1-opt.explorationDecay);
            end
          end
        end
      end

      -- loggers and plots
      local log_table = {}
      local last_trn_mse_step = {} -- for step-wsie mse
      local last_val_mse_step = {}
      local last_tst_mse_step = {}
      local last_trn_mse, last_val_mse, last_tst_mse  -- for final step mse
      for t = 1,opt.rho-1 do
        last_trn_mse_step[t]=report.optimizer.feedback['MSE_step' .. t].mse
        log_table[#log_table+1] = last_trn_mse_step[t]
      end
      last_trn_mse=report.optimizer.feedback.MSE_finstep.mse
      log_table[#log_table+1] = last_trn_mse

      if opt.lr_style=="step_if_stagnant" then
        if report.epoch==1 or last_trn_mse < best_trn_loss[report.epoch-1] then
          best_trn_loss[report.epoch] = last_trn_mse
        else
          best_trn_loss[report.epoch] = best_trn_loss[report.epoch-1]
        end
      end

      if not opt.trainOnly then
        for t = 1,opt.rho-1 do
          last_val_mse_step[t]=valid:report().feedback['MSE_step' .. t].mse
          log_table[#log_table+1] = last_val_mse_step[t]
        end
        last_val_mse=valid:report().feedback.MSE_finstep.mse
        log_table[#log_table+1] = last_val_mse

        if not opt.noTest then
          for t = 1,opt.rho-1 do
            last_tst_mse_step[t]=tester:report().feedback['MSE_step' .. t].mse
            log_table[#log_table+1] = last_tst_mse_step[t]
          end
          last_tst_mse=tester:report().feedback.MSE_finstep.mse
          log_table[#log_table+1] = last_tst_mse
      end
      end

      mse_logger:add(log_table)
      mse_logger:plot()
      print("Updated" .. mse_logger_name)

      -- record-keeping
      if last_val_mse<best_val_mse then
        best_val_epoch=report.epoch
        best_val_mse=last_val_mse
        tst_at_best_val_mse=last_tst_mse
        trn_at_best_val_mse=last_trn_mse
      end

      -- system command
      if opt.sys_cmd and report.epoch % opt.sys_cmd_iter==0 then
        os.execute(opt.sys_cmd);
      end
      -- second system command
      if opt.sys_cmd2 and report.epoch % opt.sys_cmd2_iter == 0 then
        os.execute(opt.sys_cmd2);
      end
      if report.epoch % opt.report_res_iter==0 then
        converged=0.5  -- running
        save_best_res_snap()
      end
    end
  end,
  callback = function(model, report)
    if opt.actionsType == "actor" then
      --local action_mod_step = 0;
      if opt.trackSelActions then
        for t=1, opt.rho do
          if ra._use_action_module[t] then
             --action_mod_step = action_mod_step+1;
             local tmp = ra.action.outputs[t]
             selActionsHist[t] = selActionsHist[t] + tmp:double():histc(num_actions,1,num_actions):double();
          end
        end
      end
    end

    if opt.cutoffNorm > 0 then
      local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
      opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
    end

    if opt.monitorUpdateRatio then
      print("pre-momentum layerwise update/weight ratios");
      local params, gradparams =model:parameters();
      num_param_layers = #params;
      for i=1,num_param_layers do
        local dw=gradparams[i]:norm();
        local w=params[i]:norm();
        local sz = params[i]:size();
        print(size2str(sz) .."\t" .. " : " .. string.format("%.3f", dw) .. "/" .. string.format("%.3f", w) .. " = " .. string.format("%.3f", dw/w));
      end
    end

    --local dummy, gradparams=top:parameters();
    --for i=1,#gradparams do
    --  local dw=gradparams[i]:norm();
    --  print("baseline gradparam norm")
    --  print(dw);
    --end
    -- baseline shares learningRate with the model, but sometimes easily shoots to infinity with this setting, so gradParamClip makes sure that its norm never exceeds 2
    local moduleLocal = true -- setting maximum norm for each parameter tensor locally
    top:gradParamClip(2, moduleLocal)

    -- no momentum for the baseline, which is the only learnable part of top module
    local top_momentum_params = top:momentumGradParameters()
    for i=1,#top_momentum_params do
      top_momentum_params[i]:zero()
    end
    model:updateGradParameters(opt.momentum) -- affects gradParams

    -- apply weight decay
    if opt.weightDecay~=0 then
      if not opt.biasDecayFlag then
        local weightDecayMinDim=2;
        model:weightDecay(opt.weightDecay, weightDecayMinDim) -- does not decay biases.
      else
        model_gradParams:add(opt.weightDecay, model_params);
      end
    end
    if opt.monitorUpdateRatio then
      print("post-momentum layerwise update/weight ratios");
      local params, gradparams =model:parameters();
      num_param_layers = #params;
      for i=1,num_param_layers do
        local dw=gradparams[i]:norm();
        local w=params[i]:norm();
        local sz = params[i]:size();
        print(size2str(sz) .."\t\t" .. " : " .. string.format("%.3f", dw) .. "/" .. string.format("%.3f", w) .. " = " .. string.format("%.3f", dw/w));
      end
    end

    model:updateParameters(opt.learningRate) -- affects params
    if opt.identFeedbackFlag then
      ra.rnn.recurrentModule.modules[1].modules[2].weight=torch.eye(w_size, w_size):typeAs(ra.rnn.recurrentModule.modules[1].modules[2].weight); -- inelegant, but just setting gradients for these parameters to zero does not handle weight decay...
      ra.rnn.recurrentModule.modules[1].modules[2].bias:zero();
    end

    model:maxParamNorm(opt.maxOutNorm) -- affects params
    model:zeroGradParameters() -- affects gradParams

    if opt.monitorParamShareFlag then
      print('Checking that parameter sharing is maintained')
      -- check that parameter sharing isn't broken
      --print(ra)
      local mod_list={'combine','location','patch','action', 'reconstruct', 'rotation','vectorize'};
      for mod=1,#mod_list do
        print('Checking ' .. mod_list[mod]);
        local bn=ra[mod_list[mod]].sharedClones[1]:findModules('nn.BatchNormalization')
        if #bn>0 then
          print('Found ' .. #bn .. ' batchNorm modules');
        end
        if #ra[mod_list[mod]].sharedClones>1 then
          for i=2,#ra[mod_list[mod]].sharedClones do
            local tmp=ra[mod_list[mod]].sharedClones[1]:parameters()
            local num_param = tmp and #tmp or 0
          for j=1,num_param do
            assert(ra[mod_list[mod]].sharedClones[1]:parameters()[j]:eq(ra[mod_list[mod]].sharedClones[i]:parameters()[j]):all());
          end
          if #bn>0 then
              local newbn=ra[mod_list[mod]].sharedClones[i]:findModules('nn.BatchNormalization')
            assert(bn[1].running_mean:eq(newbn[1].running_mean):all());
            if bn[1].running_var then
                assert(bn[1].running_var:eq(newbn[1].running_var):all());
              end
              if bn[1].running_std then
                assert(bn[1].running_std:eq(newbn[1].running_std):all());
              end
            end
          end
        end
        print('Okay');
      end
      print('Okay')
    end
  end,
  feedback = feedbackTable(),
  sampler = dp.ShuffleSampler{
    epoch_size = num_trn_samples, batch_size = math.min(opt.batchSize, num_trn_samples)
  },
  mse_update = opt.mseUpdate,
  progress = opt.progress
}

if not opt.trainOnly then
  valid = dp.Evaluator{
    feedback = feedbackTable(),
    sampler = dp.Sampler{epoch_size = num_val_samples, batch_size = math.min(opt.batchSize, num_val_samples)},
    progress = opt.progress
  }
  if not opt.noTest then
  tester = dp.Evaluator{
      feedback = feedbackTable(),
      sampler = dp.Sampler{epoch_size = num_tst_samples, batch_size = math.min(opt.batchSize, num_tst_samples)},
    progress = opt.progress
  }
  else
    tester=nil
  end
end

agent.opt=opt; -- so that the options are saved together with the model for easy access later on

print("Defining experiment")
--[[Experiment]]--
xp = dp.Experiment{
  id = dp.ObjectID(tostring(opt.jobno)),
  model = agent,
  optimizer = train,
  validator = valid,
  tester = tester,
  observer = {
    ad,
    dp.FileLogger('./outputs/'),
    dp.EarlyStopper{
      save_strategy = dp.SaveModelToFile(),
      max_epochs = opt.maxTries,
      error_report={opt.trainOnly and 'optimizer' or 'validator','feedback','MSE_finstep','mse'},
      maximize = false
    }
  },
  random_seed = xp_seed or os.time(),
  max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
  require 'cutorch'
  require 'cunn'
  if opt.useDevice>-1 then
    cutorch.setDevice(opt.useDevice)
  end
  xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
  print"Agent :"
  print(agent)
end

function run_exp()
  print("Experiment starts!")
  x=os.clock();
  xp:run(ds)
end
local status, err=xpcall(run_exp, debug.traceback);
if not status then
  print(err)
  converged=0;
  print("Caught error while running experiment");
else
  converged=1;
end

if opt.sys_cmd then
  os.execute(opt.sys_cmd);
end
if opt.sys_cmd2 then
  os.execute(opt.sys_cmd2);
end

tst_at_best_val_mse = tst_at_best_val_mse or 1e10
print("Epoch no:" .. best_val_epoch .. ": (val, tst)= (" .. best_val_mse .. ", " .. tst_at_best_val_mse .. ")");

save_best_res_snap()

if opt.evaluation then
  print("================================== Seen category testing ======================================== ")
  local cmd= ("th evaluate-rva.lua --jobno " .. opt.jobno .. " --evalFlag")
  os.execute(cmd);
  print("================================== Seen category drawing ======================================== ")
  local cmd= ("th evaluate-rva.lua --jobno " .. opt.jobno .. " --drawFlag --maxDrawImages 20")
  os.execute(cmd);
end
