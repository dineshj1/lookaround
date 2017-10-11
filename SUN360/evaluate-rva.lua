require 'dp'
require 'rnn'
require 'nn'
require 'os'
require 'optim'
require 'lfs'
require 'hdf5'
cwd=lfs.currentdir();
package.path = package.path .. ';' .. cwd .. '/?.lua;'
require ('SUN360ActiveMod')
require ('IdentityFeedback')
require ('VRRegressReward')
require ('MSEFeedback')
--require ('RegressionErrorFeedback')
require ('SaveModelToFile')
require ('CircShift')
require ('OneShotRNN')
require ('DummyContainer')
require ('actorMod')
require 'sun360'
require 'image'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
--cmd:option('--xpPath', './tmp_nocuda.dat', 'path to a previously saved model')
cmd:option('--jobno', 2002581, 'path to a previously saved model')
cmd:option('--resField', '', 'name of field to update with result')
--cmd:option('--cuda', false, 'model was saved with cuda')
--cmd:option('--saveConf', false, 'whether to save old confusion matrix as an image')
cmd:option('--evalFlag', false, 'whether to re-evaluate or not')
cmd:option('--drawFlag', false, 'whether to draw or not')
cmd:option('--maxDrawImages', 100, 'how many images to draw')
cmd:option('--evalData', 'test', 'which dataset to evaluate on, if any')
--cmd:option('--dump_GT', false, 'whether or not to dump ground truth montages')
--cmd:option('--outlineSelviewsFlag', true, 'whether or not to outline the selected view')
--cmd:option('--selglimpse', 1, 'used if drawglimpses>0, as the glimpse number to draw')
cmd:option('--stochastic', 0, 'evaluate the model stochatically. Generate glimpses stochastically. Remember that training is on-policy, so this is more correct.')
cmd:option('--dataset', '', 'which dataset to use : Mnist | TranslattedMnist | etc')
cmd:option('--seed', 135351, 'random seed')
--cmd:option('--overwrite', false, 'overwrite checkpoint')
--cmd:option('--timestepback', 3, 'how many steps backward to take')
--cmd:option('--no_debug', false, 'whether in debug mode or not')
cmd:option('--rho', -1, 'num timesteps')
cmd:option('--greedy', 1, 'whether greedy flag was set or not')
cmd:option('--batch_size', -1, 'whether greedy flag was set or not')
cmd:option('--num_repeats', 10, 'how many epochs to evaluate (useful if there is some element of randomness involved in evaluation)')
cmd:option('--actionsType', '', 'actor | random | presetActions | most_salient | saliency_pdf')
--cmd:option('--randomActionsFlag', false, 'whether or not to disable the action module and replace with random actions')

-- option A (for time-based preset actions)
cmd:option('--presetActions', {torch.Tensor{0,0},torch.Tensor{-1,2},torch.Tensor{-1,2},torch.Tensor{-1,2},torch.Tensor{-1,2},torch.Tensor{-1,2},torch.Tensor{-1,2},torch.Tensor{-1,2},torch.Tensor{-1,2},torch.Tensor{-1,2},torch.Tensor{-1,2},torch.Tensor{-1,2},torch.Tensor{-1,2}}, 'if presetActionsFlag is true, actor module is never actually used. Instead the given sequence of actions is performed.')
-- option B (for elev-based preset actions)
--cmd:option('--presetActions', {torch.Tensor{1,2}, torch.Tensor{-1,2},torch.Tensor{-1,2},torch.Tensor{-1,2}}, 'if presetActionsFlag is true, actor module is never actually used. Instead the given sequence of actions is performed.')
-- option C (alternative for elev-based preset actions)
--cmd:option('--presetActions', {torch.Tensor{1,2}, torch.Tensor{1,2},torch.Tensor{-1,2},torch.Tensor{-1,2}}, 'if presetActionsFlag is true, actor module is never actually used. Instead the given sequence of actions is performed.')



cmd:option('--presetActOnElevFlag', false, 'if set, presetActions are treated as corresponding to different elevations.');
cmd:option('--randomViewAverage', false, 'sets random actions flag inside RAM, disables recurrence, then passes per-timestep output through T=1 classifier')
cmd:option('--noRecurrentFlag', false, 'whether or not to disable the recurrence (to get independent per-view judgements)')
cmd:option('--zeroStepsFlag', false, 'freeze the system so that all actions are zero')
cmd:option('--drawRecurrentFlag', false, 'whether or not to visualize recurrent feedback weights')
cmd:option('--skipParameterCheck', false, 'making sure parameter sharing is not broken')
cmd:option('--startposwise', false, 'report evalFlag results by systematically varying starting position (rather than randomly)');
--cmd:option('--reconstructorNoShare', false, 'making sure parameter sharing is not broken')
--cmd:option('--network', 'vgg_m_modelnetft10303_early', 'feature extractor to be used to represent views - vgg_m | vgg_m_modelnetft10303 | vgg_m_modelnetft10303_early')
--cmd:option('--layer', 'fc7', 'layer from which features should be extracted e.g. pool5, fc7')
cmd:option('--mini', false, 'whether or not to test using mini-sized test data (useful for debugging)')
cmd:option('--memorizeObservedViews', false, 'whether or not to directly copy observed views into output viewgrid')
cmd:option('--resultfileroot', 'results', 'name of file to print results to')
cmd:text()
local opt = cmd:parse(arg or {})
io.output():setvbuf('no');
if opt.seed~=-1 then
  torch.manualSeed(opt.seed);
end

local function isempty(s)
  return s == nil or s == ''
end

if opt.evalFlag then
  function lookForCuda()
  print ("Looking for CUDA")
  require 'cunn'
  return true
  end
  local status, err=xpcall(lookForCuda, debug.traceback);
  if status then
  print ("Found CUDA");
  opt.cuda=true;
  else
  print("Did not find CUDA");
  opt.cuda=false;
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
--if not opt.no_debug then
--  require('mobdebug').start()
--end
--home=os.getenv('HOME');
opt.xpPath=('./outputs/models/' .. opt.jobno .. ".dat");
print("Evaluating model stored at " .. opt.xpPath)
--opt.xpPath=("/home/01932/dineshj/save/" .. opt.jobno .. ".dat");
model = torch.load(opt.xpPath)
if opt.cuda then
  model=model:cuda();
  -- nn.utils.recursiveType?
end

--for a,b in pairs(model) do
--  print(a);
--end

model = model.module or model.modules[1]
if opt.rho==-1 then
  print("reusing rho")
  opt.rho=model.opt.rho
end
local num_channels=3
if opt.dataset=='' then
  print("reusing dataset")
  opt.dataset=model.opt.dataset;
end

opt.datasetname = opt.dataset


if opt.batch_size==-1 then
  print("reusing batch size")
  opt.batch_size=model.opt.batchSize
end
if model.opt then
  opt.reconstructorNoShare = not model.opt.shareReconstructorsFlag
else
  opt.reconstructorNoShare = false;
end


-- check that saved model exists
assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')

if opt.cuda then
  require 'cunn'
  if opt.seed~=-1 then
  cutorch.manualSeed(opt.seed);
  end
end

function update_result(fieldname, value)
  print ("Updating result " .. fieldname)
  -- Saving params and results
  local tmp=1
  while lfs.attributes(opt.resultfileroot .. '.dat.lock')~=nil and tmp<1e5 do
    print "Waiting to save"
    tmp=tmp+1
  end
  assert(tmp<1e5, 'could not save. file indefinitely locked.');
  os.execute('touch ' .. opt.resultfileroot .. '.dat.lock');
  if lfs.attributes(opt.resultfileroot .. '.dat')~=nil  then
    reports=torch.load(opt.resultfileroot .. '.dat')
  else
    reports={}
  end
  reports[opt.jobno]['result'][fieldname]=value;
  torch.save(opt.resultfileroot .. '.dat', reports);
  os.execute('rm ' .. opt.resultfileroot .. '.dat.lock');
end



model:training(); -- it's possible this is necessary
model:evaluate();
local ra=model:findModules('nn.ActiveMod')[1];
if opt.memorizeObservedViews then
  ra.memorizeObservedViews = true
  ra.mean_subtract_input =  ra.mean_subtract_input and ra.mean_subtract_input or model.opt.mean_subtract_input
  ra.mean_subtract_output = ra.mean_subtract_output and ra.mean_subtract_output or model.opt.mean_subtract_output
end

opt.actionsType = isempty(opt.actionsType) and model.opt.actionsType or opt.actionsType

if opt.actionsType == "actor" then
  local clones=model:findModules('nn.ActiveMod')[1].action.sharedClones
  for i=1,#clones do
    if not model.opt.action_transfer == "SoftMax" then
      if opt.stochastic <= 1 then -- if opt.stochastic is 2, then there is deliberate exploration over and above just sampling from the actor's PDF
      local tmp = clones[i]:findModules('nn.AddConstant')[1];
        tmp.constant_scalar = 0;
      else
        print("Extra stochasticity: " .. tmp.constant_scalar);
      end
    else
      local tmp=clones[i]:findModules('nn.MulConstant')[1];
      if opt.stochastic <= 1 then -- if opt.stochastic is 2, then there is deliberate exploration over and above just sampling from the actor's PDF
        tmp.constant_scalar = 1; -- set temperature to 1
      else
        print("Temperature: " .. 1/tmp.constant_scalar);
      end
    end
  end
end

if opt.actionsType == "random" then
  ra.randomActionsFlag=true;
end
local load_saliency = false;
if opt.actionsType == "most_salient" then
  load_saliency = true;
  ra.mostsalientFlag = true;
end
if opt.actionsType == "saliency_pdf" then
  load_saliency = true;
  ra.saliencypdfFlag = true;
  abort() -- NOT IMPLMENTED
end

if opt.noRecurrentFlag then
  ra._avgAtTestFlag = true;
end
if opt.zeroStepsFlag then
  ra._zeroStepsFlag = true;
end
ra.nStep = opt.rho

-- check parameter sharing isn't broken
if not opt.skipParameterCheck then
  local mod_list={'combine','location','patch','reconstruct', 'rotation','vectorize'};
  if ra.IncludeActionModule then
    mod_list[#mod_list+1]='action'
  end
  for mod=1,#mod_list do
    print('Checking ' .. mod_list[mod]);
    local bn=ra[mod_list[mod]].sharedClones[1]:findModules('nn.BatchNormalization')
    if #bn>0 then
      print('Found ' .. #bn .. ' batchNorm modules');
    end
    local num_clones = #ra[mod_list[mod]].sharedClones

    if #ra[mod_list[mod]].sharedClones>1 then
      for i=2,#ra[mod_list[mod]].sharedClones do
        local tmp=ra[mod_list[mod]].sharedClones[1]:parameters()
        num_param= tmp and # tmp or 0
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
else
  print('Skipped parameter checking');
end

if opt.drawRecurrentFlag then
  print ("saving recurrent weights visualization to " .. opt.jobno .. ".jpg");
  --print(ra.rnn.recurrentModule.modules[1].modules[2].weight)
  image.save(opt.jobno .. '.jpg', ra.rnn.feedbackModule:findModules('nn.Linear')[1].weight)
end

if opt.evalFlag or opt.drawFlag then
  assert(opt.greedy)
  print(opt.dataset)
  ds = dp[opt.dataset]{
      load_all=false,
      include_cls_labels=true,
      --pixel_input = (model. opt and model.opt.pixel_input) and model.opt.pixel_input or false,
      mean_subtract_input = (model.opt and model.opt.mean_subtract_input) and model.opt.mean_subtract_input or false,
      mean_subtract_output = (model.opt and model.opt.mean_subtract_output) and model.opt.mean_subtract_output or false,
      include_saliency_scores = load_saliency,
    mini=opt.mini,
  }
  if opt.evalData == "test" then
    ds:loadTest();
    tmp=ds:testSet():targets()._input:clone();
    num_samples=tmp:size(1);
    tmp=tmp:repeatTensor(2*opt.rho+1,1,1); -- repeat along the channels dimension...
    tmp:narrow(1,opt.rho+2,opt.rho):zero(); -- setting last rho channels to zero in preparation for the lookahead loss..
    ds:testSet():targets()._input=tmp;
    ds:testSet():targets()._view='cbw';
  elseif opt.evalData=="train" then
    ds:loadTrain()
    tmp=ds:trainSet():targets()._input:clone();
    num_samples=tmp:size(1);
    tmp=tmp:repeatTensor(2*opt.rho+1,1,1); -- repeat along the channels dimension...
    tmp:narrow(1,opt.rho+2,opt.rho):zero(); -- setting last rho channels to zero in preparation for the lookahead loss..
    ds:trainSet():targets()._input=tmp;
    ds:trainSet():targets()._view='cbw';
  else
    abort();
  end

  if model.opt.mean_subtract_input or model.opt.mean_subtract_output then
    mean_view = ds._avg_viewgrid:narrow(2,1,32):narrow(3,1,32);
    avg_viewgrid_transpose=nil
  end

  -- stochastic or deterministic
  if opt.actionsType=="actor" then
    for i=1,#ra.action.sharedClones do
        local rn = ra.action.sharedClones[i]:findModules('nn.ReinforceCategorical')[1]
        rn.stochastic = opt.stochastic > 0 and true or false  -- if opt.stochastic is 1 or 2, actions are drawn from a PDF
    end
  end

  if opt.actionsType == "presetActions" then
    ra.presetActions = opt.presetActions
    ra.presetActOnElevFlag = opt.presetActOnElevFlag
    if opt.presetActOnElevFlag then
      print("WARNING: presetActOnElevFlag set")
    end
    ra.presetActionsFlag = true
    ra.IncludeActionModule = false
    ra.action = nil
  end

  if model.opt and model.opt.presetViewpointsFlag==true then
    print('WARNING: Replacing preset view points with preset actions during test time ... (otherwise cheating)');
    ra.presetViewpoints={};
    ra.presetActions={torch.Tensor{0,0}}
    for j=2,#model.opt.presetViewpoints do
      ra.presetActions[j]=model.opt.presetViewpoints[j]-model.opt.presetViewpoints[j-1];
    end
  end

  ra.average_viewgrid = ra.average_viewgrid and ra.average_viewgrid or ds._avg_viewgrid:transpose(2,3):squeeze(1);
end

if opt.drawFlag then
  print('Drawing reconstructed montages on ' .. opt.datasetname .. '(' .. opt.evalData .. ') -- saving to outputs/images/')
  --if opt.dump_GT then
  --  print('Also dumping ground truth montages on ' .. opt.datasetname .. '(' .. opt.evalData .. ') to outputs/gt_montages/')
  --end

  local data, targets
  if opt.evalData == 'test' then
    data=ds:testSet():inputs()._input;
    targets=ds:testSet():targets()._input; -- cbw
  elseif opt.evalData == "train" then
    data=ds:trainSet():inputs()._input;
    targets=ds:trainSet():targets()._input; -- cbw
  else
    abort();
  end
  for epochno=1,1 do
    --print('Epoch'..epochno)
    local count=1;
    local num_data=math.min(data:size(1),opt.maxDrawImages);
    while count<=num_data do
      if count+opt.batch_size< num_data then
        bs=opt.batch_size
      else
        bs=num_data-count+1;
      end
      local batch_data=data:narrow(1,count,bs)
      model:forward(batch_data)
      local batch_output=ra.output;
      local batch_targets=targets:narrow(2,count,bs)
      local locations=ra.selPositions;
      local actions = ra.selActions;

      gt_montage={};
      gt_montage_drawn={}
      for t=1,opt.rho do
        montages = batch_output[t]
        montages = montages:view(montages:size(1), num_channels, ds.gridshape[2]*ds.view_snapshape[2], ds.gridshape[1]*ds.view_snapshape[1]):transpose(3,4):contiguous();
        local xy=locations[t];
        local x, y = xy:select(2,1), xy:select(2,2)
        x=x*ds.view_snapshape[2]+1;
        y=y*ds.view_snapshape[1]+1;
        --require('mobdebug').start()
        for j=1,bs do
          local recon_montage=montages[j]:view(num_channels, montages:size(3), montages:size(4)):repeatTensor(3/num_channels,1,1);
          if t==1 then
            gt_montage[j]=batch_targets[1][j]:view(num_channels,ds.gridshape[2]*ds.view_snapshape[2], ds.gridshape[1]*ds.view_snapshape[1]):repeatTensor(3/num_channels,1,1):transpose(3,2);
            gt_montage_drawn[j] = gt_montage[j]:clone()
          end
          local numel = torch.prod(torch.Tensor(torch.totable(recon_montage:size())));
          local mse=torch.norm(recon_montage-gt_montage[j],2)^2/numel;
          local l1e=torch.norm(recon_montage-gt_montage[j],1)/numel;

          if model.opt.mean_subtract_output then
            if not avg_viewgrid_transpose then
              avg_viewgrid_transpose=mean_view:transpose(2,3):repeatTensor(3/num_channels,ds.gridshape[1], ds.gridshape[2]);
            end
            if t==1 then
              gt_montage_drawn[j]=gt_montage_drawn[j]+avg_viewgrid_transpose;
            end
            recon_montage=recon_montage +avg_viewgrid_transpose;
          else
            -- dummy
          end

          image.drawRect(recon_montage,y[j],x[j],y[j]+31,x[j]+31,{color={255,0,0},lineWidth=2,inplace=true});
          image.drawRect(gt_montage_drawn[j],y[j],x[j],y[j]+31,x[j]+31,{color={255,0,0},lineWidth=2,inplace=true});
          local black_strip=torch.zeros(recon_montage:size(1), 20, recon_montage:size(3)):typeAs(recon_montage);
          recon_montage = black_strip:cat(gt_montage_drawn[j],2):cat(black_strip,2):cat(recon_montage, 2);

          image.drawText(recon_montage, "MSE:".. string.format("%.2f", mse*1000) .. ",L1E:" .. string.format("%.1f", l1e*1000) .. ",prev-act:" .. actions[t][j][1]-1 .. "," .. actions[t][j][2]-1, 3,10, {color={255,0,0},bg={255,255,255},size=1,wrap=true,inplace=true});
          image.save("outputs/images/" .. opt.jobno .. "_" .. opt.datasetname .. "_" .. opt.evalData .. string.format("%04d", count+j-1) .. "time" .. string.format("%03d", t) .. '.jpg',
          recon_montage)
            ---- DISABLE THIS BLOCK IF DRAWING VIEWS NOT DESIRED
            --local tmp=ra.selPatch[t][j];
            --if model.opt.mean_subtract_input then
            --  tmp = tmp + mean_view;
            --end
            --image.save("outputs/images/" .. opt.jobno .. "_" .. opt.datasetname .. "_" .. opt.evalData .. string.format("%04d", count+j-1) .. "time" .. string.format("%03d", t) .. '_view.jpg', tmp)
        end
      end
      count=count+bs;
    end
  end
end

if opt.evalFlag then
  print('Evaluating on ' .. opt.datasetname .. '(' .. opt.evalData .. ')')
  --ds=dp[opt.dataset]{ network = opt.network};
  data = ds:get(opt.evalData, 'inputs');
  targets = ds:get(opt.evalData, 'targets');
  if opt.evalData=="train" then
    cls_labs = ds.trn_labs;
  elseif opt.evalData=="test" then
    cls_labs = ds.tst_labs;
  else
    abort()
  end
  local saliency_scores
  if load_saliency then
    print("Loading saliency")
    assert(opt.evalData=="test")
    saliency_scores = ds.tst_sal_scores;
  end
  classnames={
    'clsname1',
    'clsname2',
    'clsname3',
    'clsname4',
    'clsname5',
    'clsname6',
    'clsname7',
    'clsname8',
    'clsname9',
    'clsname10',
    'clsname11',
    'clsname12',
    'clsname13',
    'clsname14',
    'clsname15',
    'clsname16',
    'clsname17',
    'clsname18',
    'clsname19',
    'clsname20',
    'clsname21',
    'clsname22',
    'clsname23',
    'clsname24',
    'clsname25',
    'clsname26'
  }
  local sse=torch.zeros(opt.rho):totable();
  local s1e=torch.zeros(opt.rho):totable();
  local clswise_sse = torch.zeros(opt.rho,26):totable();
  local clswise_s1e = torch.zeros(opt.rho,26):totable();
  local cls_ct = torch.zeros(26):totable();
  --ensemble_correct=0;
  local minAct=ra._action_values[1]:min()
  local maxAct=ra._action_values[1]:max()
  local num_actions = model.opt.action_gridshape[1]*model.opt.action_gridshape[2];
  local min_act_val = (ra._action_values[1]:min())*model.opt.action_gridshape[2]+ra._action_values[2]:min()
  local max_act_val = min_act_val+num_actions
  local jointSelActionsHist={}
  local selActionsHist={}
  selActionsHist[1]={}
  selActionsHist[2]={}
  local num_data
  num_start_pos = opt.startposwise and ds.gridshape[1]*ds.gridshape[2] or 1;

  num_repeats=opt.num_repeats;
  mse = {};
  l1e = {}
  clswise_mse ={};
  clswise_l1e ={};
  for startpos_ind=0,num_start_pos-1 do
    if opt.startposwise then
      start_rowno=torch.floor(startpos_ind/ds.gridshape[2]);
      start_colno=startpos_ind - start_rowno*ds.gridshape[2];
      ra.presetActionsFlag = false;
      ra.presetViewpointsFlag=true;
      ra.presetViewpoints={torch.Tensor{start_rowno,start_colno}}; -- makes ActiveMod module pick the same starting position each time
      print(ra.presetViewpoints[1])
      num_repeats=1;
      --mse={};
      --clswise_mse={};
      clswise_sse = torch.zeros(opt.rho,26):totable();
      clswise_s1e = torch.zeros(opt.rho,26):totable();
      cls_ct = torch.zeros(26):totable();
      sse=torch.zeros(opt.rho):totable();
      s1e=torch.zeros(opt.rho):totable();
    end
    for epochno=1,num_repeats do
      if num_repeats>1 then
      print('Epoch'..epochno)
      end
      local count=0;
      num_data=data:size(1);
      while count<num_data do
        local bs
        if count+opt.batch_size< num_data then
          bs=opt.batch_size
        else
            bs=num_data-count;
        end
        local batch_data=data:narrow(1,count+1,bs)
        if load_saliency then
          ra.saliency_viewgrid = saliency_scores:narrow(1,count+1,bs)
        end
        model:forward(batch_data)
        local batch_output = ra.output;
        local batch_targets=targets:narrow(2,count+1,bs)
        local batch_cls_labs = cls_labs:narrow(1,count+1,bs);
        local dummy
        batch_recon={}
        for t=1,opt.rho do
          -- record data for mse computation
          local diff=batch_output[t]:clone();
          diff:csub(batch_targets[t]:typeAs(diff));
          local l2err=diff:norm(2,2):pow(2)/diff:size(2)
          local l1err=diff:norm(1,2)/diff:size(2)
          sse[t]=sse[t]+l2err:sum();
          s1e[t]=s1e[t]+l1err:sum();

          -- update clswise_sse
          for j=1,bs do
            clswise_sse[t][batch_cls_labs[j]+1]=clswise_sse[t][batch_cls_labs[j]+1]+l2err[j][1];
            clswise_s1e[t][batch_cls_labs[j]+1]=clswise_s1e[t][batch_cls_labs[j]+1]+l1err[j][1];
          end

          -- record action choices
          local tmp=(ra.selActions[t]-1):select(2,1):float():histc(maxAct-minAct+1, minAct, maxAct)
          selActionsHist[1][t]=selActionsHist[1][t] and selActionsHist[1][t]+tmp or tmp
          local tmp=(ra.selActions[t]-1):select(2,2):float():histc(11,-5.5,5.5)
          selActionsHist[2][t]=selActionsHist[2][t] and selActionsHist[2][t]+tmp or tmp

          if opt.actionsType== "actor" then
            if ra._use_action_module[t] then
              local tmp_hist = ra.action.sharedClones[t].output:double():histc(num_actions, 1, num_actions):double()
              jointSelActionsHist[t] = jointSelActionsHist[t] and jointSelActionsHist[t]+tmp_hist or tmp_hist;
            end
          else
            local tmp = ((ra.selActions[t]:select(2,1)-1)*model.opt.action_gridshape[2]+ra.selActions[t]:select(2,2)-1):double();
            --local tmp_hist = tmp:histc(num_actions,1,num_actions):double();
            local tmp_hist = tmp:histc(num_actions,min_act_val,max_act_val):double();
            jointSelActionsHist[t] = jointSelActionsHist[t] and jointSelActionsHist[t]+tmp_hist or tmp_hist;
          end
        end
        for j=1,bs do
          cls_ct[batch_cls_labs[j]+1]=cls_ct[batch_cls_labs[j]+1]+1;
        end
        --dummy, batch_ensemblepreds=torch.max(ensemble_output, 2)
        --ensemble_correct=ensemble_correct+batch_labels:cuda():eq(batch_ensemblepreds):sum()
        count=count+bs
      end
    end
    if not opt.startposwise then
      -- compute and print MSE
      --mse = {}
      --clswise_mse = {}
      print("Classwise MSE (CSV format)");
      io.write("classname, #samples, ")
      for t=1,opt.rho do
        if t>1 then
          io.write("% improvement at t=" .. t);
        end
        io.write("MSE*1000 at t=" .. t);
        clswise_mse[t]={}
        clswise_l1e[t]={}
      end
          io.write("\n")
      for j=1,26 do -- 40
        if cls_ct[j]>0 then
          for t=1,opt.rho do
            clswise_mse[t][j]=clswise_sse[t][j]/cls_ct[j];
            clswise_l1e[t][j]=clswise_s1e[t][j]/cls_ct[j];
            if t==1 then
               io.write(classnames[j] .. ", " .. cls_ct[j] ..  ", ");
            else
              io.write(", ");
              local tmp=(clswise_mse[t][j]-clswise_mse[t-1][j])/clswise_mse[t-1][j]*100;
              if tmp>0 then
                io.write("+")
              end
              io.write(string.format("%.1f", tmp)  .. ", ");
            end
            io.write(string.format("%.2f", clswise_mse[t][j]*1000));
            io.write(string.format("[l1:%.2f]", clswise_l1e[t][j]*1000));
            if t==opt.rho then
              io.write("\n");
            end
          end
        end
      end
      for t=1,opt.rho do
        mse[t]=sse[t]/(num_data*num_repeats)
        l1e[t]=s1e[t]/(num_data*num_repeats)
        print("Overall MSE (x1000) at timestep ".. t .. ", " .. num_data*num_repeats .. ", " .. string.format("%.2f", mse[t]*1000));
        print("Overall L1E (x1000) at timestep ".. t .. ", " .. num_data*num_repeats .. ", " .. string.format("%.2f", l1e[t]*1000));
        if opt.resField~='' then
          update_result(opt.resField .. t, mse[t]);
        end
        --clswise_mse[t]={};
      end
      --if model.opt.actionsType=="actor" then
      print("Joint histograms")
      for t=2,opt.rho do
        local tmp =jointSelActionsHist[t];
        --tmp = tmp/tmp:sum();
        print(tmp:reshape(model.opt.action_gridshape[1], model.opt.action_gridshape[2]));
      end
    else
      --print(start_rowno)
      --print(start_colno)
      if start_colno == 0 then
        mse[start_rowno+1]= {}
        l1e[start_rowno+1]= {}
        clswise_mse[start_rowno+1]={}
        clswise_l1e[start_rowno+1]={}
      end
      --print(mse[start_rowno])
      mse[start_rowno+1][start_colno+1]={};
      l1e[start_rowno+1][start_colno+1]={};
      clswise_mse[start_rowno+1][start_colno+1]={};
      clswise_l1e[start_rowno+1][start_colno+1]={};
      for t=1,opt.rho do
        clswise_mse[start_rowno+1][start_colno+1][t]={};
        clswise_l1e[start_rowno+1][start_colno+1][t]={};
      end
      for j=1,26 do
        if cls_ct[j]>0 then
          for t=1,opt.rho do
            clswise_mse[start_rowno+1][start_colno+1][t][j] = clswise_sse[t][j]/cls_ct[j];
            clswise_l1e[start_rowno+1][start_colno+1][t][j] = clswise_s1e[t][j]/cls_ct[j];
          end
        else
          for t=1,opt.rho do
            clswise_mse[start_rowno+1][start_colno+1][t][j] = -1;
            clswise_l1e[start_rowno+1][start_colno+1][t][j] = -1;
          end
        end
      end
      for t=1,opt.rho do
        mse[start_rowno+1][start_colno+1][t]=sse[t]/(num_data*num_repeats)
        l1e[start_rowno+1][start_colno+1][t]=s1e[t]/(num_data*num_repeats)
        print("Overall MSE (x1000) at timestep ".. t .. ", " .. num_data*num_repeats .. ", " .. mse[start_rowno+1][start_colno+1][t]*1000);
        print("Overall L1E (x1000) at timestep ".. t .. ", " .. num_data*num_repeats .. ", " .. l1e[start_rowno+1][start_colno+1][t]*1000);
      end
    end
  end
  if opt.startposwise then
    mse=torch.Tensor(mse);
    clswise_mse = torch.Tensor(clswise_mse);
    -- print MSE chart by starting position for every time
    for t=1,opt.rho do
      if opt.rho>1 then
        print("Time " .. t);
      end
      print("MSE chart by position");
      local tmp = mse:select(3,t)
      --print(tmp);
      local myfilename = "outputs/" .. opt.jobno .. "-" .. opt.dataset .. (opt.mini and "mini" or "") .. (opt.unseenCategoryTest and "unseen" or "seen") .. "-poswiseMSE.h5"
      print("Saving to " .. myfilename);
      local myfile=hdf5.open(myfilename, "w");
      myfile:write("/overall", tmp);
      -- print MSE chart by starting position, clswise
      for j=1,26 do
        if cls_ct[j]>0 then
          print(classnames[j] .. " class MSE chart by position");
          local tmp=clswise_mse:select(3,t)
          tmp=tmp:select(3,j)
          --print(tmp);
          myfile:write("/" .. classnames[j] , tmp);
          --torch.save(opt.jobno .. "-" .. opt.dataset  .. (opt.mini and "mini" or "") .. (opt.unseenCategoryTest and "unseen_" or "seen_") .. classnames[j] .. "-poswiseMSE.dat", tmp)
        end
      end
      myfile:close();
    end
  end
end

if opt.randomViewAverage then
  abort()
  print('Evaluating on ' .. opt.datasetname .. '(' .. opt.evalData .. ')')
  ds=dp[opt.dataset]{
    network=opt.network,
    layer=opt.layer,
    load_all=false,
    include_cls_labels=true,
    only_test=true,
    pixel_input = (model. opt and model.opt.pixel_input) and model.opt.pixel_input or false,
    mini=opt.mini,
  };
  ra.average_viewgrid = ra.average_viewgrid and ra.average_viewgrid or ds._avg_viewgrid:transpose(2,3):squeeze(1);
  data = ds:get(opt.evalData, 'inputs');
  targets = ds:get(opt.evalData, 'targets', 'b');
  correct=torch.zeros(opt.rho):totable();
  local ensemble_correct=torch.zeros(opt.rho):totable();
  ra._avgAtTestFlag = true; --disabling recurrence
  ra.randomActionsFlag=true --disabling intelligent action selection
  reconstructor_array=model.modules[3].modules[1].modules[1];
  one_view_reconstructor=reconstructor_array.modules[1].modules[3]; -- time step 1, and skipping table selection
  --local minAct=ra._action_values[1]:min()
  --local maxAct=ra._action_values[1]:max()
  local ensemble_output={};
  local batch_ensemblepreds={};
  for epochno=1,opt.num_repeats do
    print('Epoch'..epochno)
    count=1;
    num_data=data:size(1);
    while count<=num_data do
      if count+opt.batch_size< num_data then
        bs=opt.batch_size
      else
        bs=num_data-count+1;
      end
      batch_data=data:narrow(1,count,bs)
      batch_origOutput=model:forward(batch_data)
      batch_targets=targets:narrow(1,count,bs)
      local dummy
      batch_output={}
      batch_predlabels={}
      for t=1,opt.rho do
      -- record accuracies
        batch_output[t]=one_view_reconstructor:forward(ra.output[t][1]);
        dummy, batch_predlabels[t]=torch.max(batch_output[t], 2)
        correct[t]=correct[t]+batch_labels:cuda():eq(batch_predlabels[t]):sum()
        ensemble_output[t]= t==1 and batch_output[t]:clone() or ensemble_output[t-1]+batch_output[t]
        dummy, batch_ensemblepreds[t]=torch.max(ensemble_output[t], 2)
        ensemble_correct[t]=ensemble_correct[t]+batch_labels:cuda():eq(batch_ensemblepreds[t]):sum()
      end
      --print(correct)
      --print(ensemble_correct)
      --print(batch_output[1][1])
      --print(batch_output[2][1])
      --print(batch_output[3][1])
      --print(ensemble_output[1][1])
      --print(ensemble_output[2][1])
      --print(ensemble_output[3][1])
      count=count+bs
    end
  end
  --print(ra.selActions[2]-1)
  --print(maxAct)
  --abort()
  accuracy={}
  for t=1,opt.rho do
    accuracy[t]=correct[t]/(num_data*opt.num_repeats)
    print("Timestep ".. t .. ":" .. accuracy[t])

    if opt.resField~='' then
      update_result(opt.resField .. t, accuracy[t]);
    end
  end
  print("Ensemble accuracy (averaging over views):");
  local ensemble_acc={}
  for t=1, opt.rho do
    print('Time '.. t .. ': ');
    ensemble_acc[t]=ensemble_correct[t]/(num_data*opt.num_repeats)
    print(ensemble_acc[t])
  end
  if opt.resField~='' then
    update_result(opt.resField .. 'avg', ensemble_acc);
  end
end
