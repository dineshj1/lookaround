-----------------------------------------------------------------------
--[[ ActiveMod ]]--
-- Ref. A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf
-- module which takes an RNN as argument with other
-- hyper-parameters such as the maximum number of steps,
-- action (actions sampling module like ReinforceNormal) and
------------------------------------------------------------------------
local ActiveMod, parent = torch.class("nn.ActiveMod", "nn.AbstractSequencer")

function ActiveMod:__init(config)
  config = config or {}
  assert(torch.type(config) == 'table' and not config[1], "Constructor requires key-value arguments")
  local args
  args, self.action_lr_mult, self.rnn_lr_mult, self.decoder_lr_mult, self.finetuneFactor, self.rnn, self.location, self.patch, self.combine, self.action, self.reconstruct, self.rotation, self.vectorize, self.nStep, self._view_gridshape, self._action_gridshape, self._glimpseshape, self._viewshape, self.pretrainModeIters, self.randomActionsFlag, self.presetViewpoints, self.presetActions, self.presetActOnElevFlag, self._startAnywhereFlag, self._action_values, self._wrapAroundFlag, self._avgAtTestFlag, self._zeroStepsFlag, self._zeroFirstStepFlag, self.actOnPixels, self.actOnTime, self.actOnElev, self.actOnAzim, self.knownElev, self.knownAzim, self.rotationCompensationFlag, self.compensateKnownPosFlag, self.memorizeObservedViews, self.mean_subtract_input, self.mean_subtract_output, self.average_viewgrid, self.ignoreLocationFlag = xlua.unpack(
     {config},
     'ActiveMod',
     '',
     {arg='action_lr_mult', type='number', default=1},
     {arg='rnn_lr_mult', type='number', default=1},
     {arg='decoder_lr_mult', type='number', default=1},
     {arg='finetuneFactor', type='number', default=1, help='factor by which all modules trained in rho=1 are multiplied ... useful when finetuning from a rho=1-pretrained model'},
     {arg='rnnMod', type='nn.AbstractRecurrent', help='simple recurrent module'},
     {arg='locationMod', type='nn.Module'},
     {arg='patchMod', type='nn.Module'},
     {arg='combineMod', type='nn.Module'},
     {arg='actionMod', type='nn.Module'},
     {arg='reconstructMod', type='nn.Module'},
     {arg='rotateMod', type='nn.Module'},
     {arg='vectorizeMod', type='nn.Module'},
     {arg='nStep', type='number', default=1},
     {arg='view_gridshape', type='table', default=torch.IntTensor{8,12}},
     {arg='action_gridshape', type='table'},
     --{arg='action_grid_origin', type='table'},
     {arg='glimpseshape', type='table', default=torch.IntTensor{16,16}},
     {arg='viewshape', type='table', default=torch.IntTensor{32,32}},
     {arg='pretrainModeIters', type='number', default=0},
     {arg='randomActionsFlag', type='boolean', default=false},
     {arg='presetViewpoints', type='table', default={}},
     {arg='presetActions', type='table', default={}},
     {arg='presetActOnElevFlag', type='boolean', default=false, help = "if set, the indices of the presetActions table will be treated as elevations"},
     {arg='startAnywhereFlag', type='boolean', default=false},
     {arg='action_values', type='table', default={}},
     {arg='wrapAroundFlag', type='table', default={false,true}},
     {arg='avgAtTestFlag', type='boolean', default=false, 'if true, then at test time, recurrent module will be skipped, so that in effect, only per instant features are classified'},
     {arg='zeroStepsFlag', type='boolean', default=false, 'if true, all actions will be zero at test time'},
     {arg='zeroFirstStepFlag', type='boolean', default=false, 'if true, first view will be completely beyond control'},
     {arg='actOnPixels', type='boolean', default=false},
     {arg='actOnTime', type='boolean', default=true, 'if true, actor module will receive time as input'},
     {arg='actOnElev', type='boolean', default=false, 'if true, actor module processes absolute elevation (only possible if knownElev)'},
     {arg='actOnAzim', type='boolean', default=false, 'if true, actor module processes absolute azimuth (only possible if knownAzim)'},
     {arg='knownElev', type='boolean', default=false, 'if true, locator module processes absolute elevation'},
     {arg='knownAzim', type='boolean', default=false, 'if true, locator module processes absolute azimuth'},
     {arg='rotationCompensationFlag', type='boolean', default=false, 'if true, viewgrid reconstructions will be shifted to the starting position'},
     {arg='compensateKnownPosFlag', type='boolean', default=false, 'if false, then viewgrid reconstructions will not be shifted along coordinates that are known. e.g. if knownElev, then the network is expected to learn to use that to produce viewgrids that are correctly registered with target viewgrid in elevation.'},
     {arg='memorizeObservedViews', type='boolean', default=false, 'stores observed views exactly in memory'},
     {arg='mean_subtract_input', type='boolean', default=false, 'for memorizeObservedViews'},
     {arg='mean_subtract_output', type='boolean', default=false, 'for memorizeObservedViews'},
     {arg='average_viewgrid', default={}, 'for memorizeObservedViews'},
     {arg='ignoreLocationFlag', type='boolean', default=false, 'if true, no location sensor'}
  )
  parent.__init(self)

  self.mostsalientFlag = false
  self.saliencypdfFlag= false

  self._iter_count=0;

  self.efficientEval = false; --can set this to true in exceptional circumstances, when, at evaluation time, the desired behavior is to forget all stepwise outputs

  if not self._action_gridshape then
    self._action_gridshape=self._view_gridshape;
  end

  if next(self.presetViewpoints)==nil then
    self.presetViewpointsFlag = false
  else
    self.presetViewpointsFlag = true
  end

  if next(self.presetActions)==nil then
    self.presetActionsFlag = false
  else
    self.presetActionsFlag = true
  end

  self.rnn.copyInputs = true -- determines whether inputs are copied or set

  self.IncludeActionModule = false
  if not (self.randomActionsFlag or (self.presetViewpointsFlag and #self.presetViewpoints>=self.nStep) or (self.presetActionsFlag and #self.presetActions>=self.nStep) or self._zeroStepsFlag) then
    self.IncludeActionModule = true
  end

  self.modules={}

  if self.IncludeActionModule then
    self.action = (not torch.isTypeOf(self.action, 'nn.AbstractRecurrent')) and nn.Recursor(self.action) or self.action
    self.modules[#self.modules+1]=self.action;
  end
  self.patch = (not torch.isTypeOf(self.patch, 'nn.AbstractRecurrent')) and nn.Recursor(self.patch) or self.patch
  self.modules[#self.modules+1]=self.patch;
  if not self.ignoreLocationFlag then
    self.location = (not torch.isTypeOf(self.location, 'nn.AbstractRecurrent')) and nn.Recursor(self.location) or self.location
    self.modules[#self.modules+1]=self.location;
    self.combine = (not torch.isTypeOf(self.combine, 'nn.AbstractRecurrent')) and nn.Recursor(self.combine) or self.combine
    self.modules[#self.modules+1]=self.combine;
  end
  self.rnn = (not torch.isTypeOf(self.rnn, 'nn.AbstractRecurrent')) and nn.Recursor(self.rnn) or self.rnn
  self.modules[#self.modules+1]=self.rnn;
  self.reconstruct = (not torch.isTypeOf(self.reconstruct, 'nn.AbstractRecurrent')) and nn.Recursor(self.reconstruct) or self.reconstruct
  self.modules[#self.modules+1]=self.reconstruct;
  if self.rotationCompensationFlag  then
    self.rotation = (not torch.isTypeOf(self.rotation, 'nn.AbstractRecurrent')) and nn.Recursor(self.rotation) or self.rotation
    self.modules[#self.modules+1]=self.rotation;
  end
  self.vectorize = (not torch.isTypeOf(self.vectorize, 'nn.AbstractRecurrent')) and nn.Recursor(self.vectorize) or self.vectorize
  self.modules[#self.modules+1]=self.vectorize;

  self.output = {} -- rnn output
  self.selActions = {} -- selected actions
  self.selPositions = {} -- selected positions
  self.combineOutputs = {}
  self.locationOutputs = {}
  self.reconstructOutputs = {}
  self.rotated_reconstructOutputs = {}
  self.vectorized_reconstructOutputs = {}
  self.selPatch={}
  self.patchOutputs = {}
  self.rnnOutputs = {}
  self.actionInputs = {}
  self.locationInputs = {}

  self.gradInput_vectorize = {}
  self.gradInput_rotation = {}
  self.gradInput_reconstruct = {}
  self.gradInput_combine = {}
  self.gradOutput_patch = {}
  self._use_action_module = {}
end

function ActiveMod:updateOutput(input) -- forward computation
  if self.train then
    self._iter_count=self._iter_count+1; -- for enforcing self.pretrainModeIters
  end
  for i = 1,#self.modules do
    self.modules[i]:forget()  -- forget all past inputs, and set internal step counter to 1
  end
  if self.train and self._iter_count==self.pretrainModeIters+1 then
    print "--------------------------------"
    print "proper training begins ..."
    print "--------------------------------"
  end
  local batch_sz=input:size(1);
  for step=1,self.nStep do
    -- ACTION SELECTION
    self._use_action_module[step] = false
    local curr_action_op
    -- PRODUCING SELACTIONS[STEP], SELPOSITIONS[STEP]
    if self.presetViewpointsFlag and self.presetViewpoints[step] then
        self.selPositions[step]=input.new():resize(batch_sz,2):zero();
        local pos_rowno = self.selPositions[step]:select(2,1);
        pos_rowno:zero():add(self.presetViewpoints[step][1]);
        local pos_colno = self.selPositions[step]:select(2,2);
        pos_colno:zero():add(self.presetViewpoints[step][2]);
        if step>1 then
          self.selActions[step]=self.selPositions[step]-self.selPositions[step-1];
        else
          self.selActions[step]=self.selPositions[step].new():resize(batch_sz,2):zero();
        end
    else
      local start_a, start_b, step_a, step_b;

      -- PRODUCING START_A and START_B
      self.selPositions[step]=input.new():resize(batch_sz,2);
      if step>1 then
        start_a=self.selPositions[step-1]:select(2,1);
        start_b=self.selPositions[step-1]:select(2,2);
      else
        if not self._startAnywhereFlag then
          -- reference starting point on first step is grid center
          start_a=torch.ceil(self._view_gridshape[1]/2)
          start_b=torch.ceil(self._view_gridshape[2]/2)
        else
          start_a=torch.zeros(batch_sz):random(0,self._view_gridshape[1]-1):typeAs(input)
          start_b=torch.zeros(batch_sz):random(0,self._view_gridshape[2]-1):typeAs(input)
        end
      end

      -- PRODUCING STEP_A, STEP_B, SELACTIONS
      if self.mostsalientFlag and not self.train then
        assert(self.saliency_viewgrid) -- must be externally set
        step_a = input.new():resize(batch_sz,1):zero()
        step_b = input.new():resize(batch_sz,1):zero()
        if step==1 then -- no action at first time instant
          step_a:add(1)
          step_b:add(1)
          self.saliency_map_padded = false -- NOTE: assumes that self.saliency_viewgrid has been manually set externally before passing in new minibatch
        else
          if not self.saliency_map_padded then
            self.saliency_padding_a = torch.floor(self._action_gridshape[1]/2)
            self.saliency_padding_b = torch.floor(self._action_gridshape[2]/2)
            self.saliency_viewgrid = nn.Padding(1,-self.saliency_padding_a,2,-200):forward(self.saliency_viewgrid) -- negative padding along dim 1
            self.saliency_viewgrid = nn.Padding(1,self.saliency_padding_a,2,-200):forward(self.saliency_viewgrid) -- positive padding along dim 1
            self.saliency_viewgrid = nn.Padding(2,-self.saliency_padding_b,2,-200):forward(self.saliency_viewgrid) -- negative padding along dim 2
            self.saliency_viewgrid = nn.Padding(2,self.saliency_padding_b,2,-200):forward(self.saliency_viewgrid) -- positive padding along dim 2
            self.saliency_map_padded = true
            -- should produce 6x12 map from input 4x8 map, with 3x5 action grid
          end
          for sampleIdx=1,batch_sz do -- each sample in batch
            local sample_sal_viewgrid = self.saliency_viewgrid[sampleIdx]
            local sample_start_a = start_a[sampleIdx] -- indexing starts from 0
            local sample_start_b = start_b[sampleIdx] -- indexing starts from 0
            --print(sample_start_a .. ", " .. sample_start_b)
            local sample_sal_nbd = sample_sal_viewgrid:narrow( -- remember this has been padded
              1,
              sample_start_a + 1,self._action_gridshape[1]):narrow( -- remember this has been padded
              2,
              sample_start_b + 1,self._action_gridshape[2]);
            -- Cannot select zero action, so set center element to zero
            sample_sal_nbd[self.saliency_padding_a+1][self.saliency_padding_b+1]=-200;
            -- Now pick out largest element in sample_sal_nbd
            local tmp1, tmp2 = sample_sal_nbd:max(1)
            tmp2=tmp2:squeeze()
            local tmp3, tmp4 = tmp1:max(2)
            tmp4=tmp4:squeeze()
            step_b[sampleIdx] = tmp4
            --print(tmp2[tmp4])
            --print(tmp4)
            step_a[sampleIdx] = tmp2[tmp4]
            step_a[sampleIdx] = step_a[sampleIdx] - self.saliency_padding_a
            step_b[sampleIdx] = step_b[sampleIdx] - self.saliency_padding_b
            --print(step_a[sampleIdx])
            --print(step_b[sampleIdx])
          end
        end
        self.selActions[step]=nn.JoinTable(2):forward{
            step_a:contiguous():view(step_a:size(1),1),
            step_b:contiguous():view(step_b:size(1),1),
          }:typeAs(input);
      elseif self.presetActionsFlag and (self.presetActions[step] or self.presetActOnElevFlag) then
        if self.presetActOnElevFlag then
          -- presetActions table indices are taken to correspond to elevation
          step_a = input.new():resize(batch_sz,1):zero()
          step_b = input.new():resize(batch_sz,1):zero()
          if step==1 then -- no action at first time instant
            step_a:add(1)
            step_b:add(1)
          else
            for sampleIdx=1,batch_sz do -- each sample in batch
              local elev = start_a[sampleIdx]+1
              step_a[sampleIdx]=self.presetActions[elev][1]+1
              step_b[sampleIdx]=self.presetActions[elev][2]+1
            end
          end
        else
          -- presetActions table indices are taken to correspond to time
          step_a = input.new():resize(batch_sz,1):zero():add(self.presetActions[step][1]+1);
          step_b = input.new():resize(batch_sz,1):zero():add(self.presetActions[step][2]+1);
        end
        self.selActions[step]=nn.JoinTable(2):forward{
            step_a:contiguous():view(step_a:size(1),1),
            step_b:contiguous():view(step_b:size(1),1),
          }:typeAs(input);
      elseif (self._zeroFirstStepFlag and step==1) or ((not self.train) and self._zeroStepsFlag) then
        step_a = input.new():resize(batch_sz,1):zero():add(1);
        step_b = input.new():resize(batch_sz,1):zero():add(1);
        self.selActions[step]=nn.JoinTable(2):forward{
          step_a:contiguous():view(step_a:size(1),1),
          step_b:contiguous():view(step_b:size(1),1),
        }:typeAs(input);
      else
        if self.randomActionsFlag or (self._iter_count<=self.pretrainModeIters) then --override previously selected action
          curr_action_op=input.new():resize(batch_sz,1):type('torch.DoubleTensor'):random(1,self._action_gridshape[1]*self._action_gridshape[2]):typeAs(input);
        else
          self._use_action_module[step]=true
          local action_inputsz = self.action:parameters()[1]:size(2);
          if step == 1 then
            -- sample an initial starting action by forwarding zeros through action module
            self._initActionInput = self._initActionInput or input.new()
            self._initActionInput:resize(batch_sz,action_inputsz):zero()
            if self.actOnPixels then
              local zeropatch=input.new();
              zeropatch[step]:resize(batch_sz, input:size(2), self._glimpseshape[2], self._glimpseshape[1])
              self._initActionInput = {self._initActionInput, zeropatch}
            end
            curr_action_op = self.action:updateOutput(self._initActionInput):resize(batch_sz,1);
          else
            self.actionInputs[step]=self.rnnOutputs[step-1]:clone();
            self.actionInputs[step]=self.actionInputs[step]:cat(self.selPositions[step-1]-self.selPositions[1]) -- relative elev and azim
            if self.actOnTime then
              self.actionInputs[step]=self.actionInputs[step]:cat(torch.ones(batch_sz):typeAs(input)*step/10, 2);
            end
            if self.knownElev==true and self.actOnElev==true then --absolute elev
              self.actionInputs[step]=self.actionInputs[step]:cat(self.selPositions[step-1]:narrow(2,1,1));
            end
            if self.knownAzim==true and self.actOnAzim==true then --absolute azim
              self.actionInputs[step]=self.actionInputs[step]:cat(self.selPositions[step-1]:narrow(2,2,1));
            end
            if self.actOnPixels then
              self.actionInputs[step]={self.actionInputs[step], self.selPatch[step-1]}
            end
            curr_action_op = self.action:updateOutput(self.actionInputs[step]):resize(batch_sz,1);
          end
        end
        -- SELECTING PATCH BASED ON ACTION
        local colno=curr_action_op:typeAs(input);
        local rowno=torch.floor((colno-1)/self._action_gridshape[2]); -- index starting from zero
        colno:add(-rowno*self._action_gridshape[2]);
        colno=colno-1; -- index starting from zero
        assert(next(self._action_values));
        step_a=self._action_values[1]:index(1,(rowno:type('torch.LongTensor')+1):view(rowno:size(1)))+1;
        step_b=self._action_values[2]:index(1,(colno:type('torch.LongTensor')+1):view(colno:size(1)))+1;
        self.selActions[step]=nn.JoinTable(2):forward{
          step_a:view(step_a:size(1),1),
          step_b:view(step_b:size(1),1),
        }:typeAs(rowno);
      end


      -- PRODUCING SELPOSITIONS[STEP]
      self.selPositions[step]:select(2,1):copy(step_a+start_a-1);
      self.selPositions[step]:select(2,2):copy(step_b+start_b-1);
      local pos_rowno=self.selPositions[step]:narrow(2,1,1);
      local pos_colno=self.selPositions[step]:narrow(2,2,1);
      if torch.type(self._wrapAroundFlag)=='boolean' then -- for backward compatibility
        self._wrapAroundFlag={self._wrapAroundFlag, self._wrapAroundFlag};
      end
      -- colno (latitude)
      if self._wrapAroundFlag[2] then
        -- simple modulus
        local tmp=torch.floor(pos_colno/self._view_gridshape[2])*self._view_gridshape[2];
        pos_colno:add(-tmp);
        assert(torch.min(pos_colno)>=0)
      else
        -- capping at boundaries
        pos_colno[torch.gt(pos_colno, self._view_gridshape[2]-1)]=self._view_gridshape[2]-1;
        pos_colno[torch.lt(pos_colno, 0)]=0;
      end
      -- rowno (altitude)
      if self._wrapAroundFlag[1] then
        -- simple modulus (input should include latitudes spanning 360 degrees)
        local tmp=torch.floor(pos_rowno/self._view_gridshape[1])*self._view_gridshape[1];
        pos_rowno:add(-tmp);
        assert(torch.min(pos_rowno)>=0)
      else
        -- capping at boundaries
        pos_rowno[torch.gt(pos_rowno, self._view_gridshape[1]-1)]=self._view_gridshape[1]-1;
        pos_rowno[torch.lt(pos_rowno, 0)]=0;
      end
    end
    if self.IncludeActionModule and self._use_action_module[step]==false then
      self.action.step= self.action.step+1 -- to avoid problems with multi-timestep reinforce()
    end


    -- PRODUCING SELPATCH[STEP]
    self.selPatch[step]=input.new();
    self.selPatch[step]:resize(batch_sz, input:size(2), self._glimpseshape[2], self._glimpseshape[1])
    for sampleIdx=1,batch_sz do -- each sample in batch
      local inputSample = input[sampleIdx] -- C*H*W
      self.selPatch[step][sampleIdx]=inputSample:narrow(2, self.selPositions[step][sampleIdx][1]*self._glimpseshape[1]+1, self._glimpseshape[1]):narrow(3, self.selPositions[step][sampleIdx][2]*self._glimpseshape[2]+1, self._glimpseshape[2]);
    end
    self.patchOutputs[step]=self.patch:updateOutput(self.selPatch[step]);

    -- PRODUCING COMBINEOUTPUTS[STEP]
    if not self.ignoreLocationFlag then
      self.locationInputs[step]=(self.selPositions[step]-self.selPositions[1]);  --TODO divide by 10
      self.locationInputs[step]=self.locationInputs[step]:cat(torch.ones(batch_sz):typeAs(input)*1, 2); -- was earlier supposed to represent current time instance. removed now.
      if self.knownElev then
        self.locationInputs[step]=self.locationInputs[step]:cat(self.selPositions[step]:narrow(2,1,1));
      end
      if self.knownAzim then
        self.locationInputs[step]=self.locationInputs[step]:cat(self.selPositions[step]:narrow(2,2,1));
      end
      self.locationOutputs[step]=self.location:updateOutput(self.locationInputs[step]);
      self.combineOutputs[step]=self.combine:updateOutput{self.locationOutputs[step], self.patchOutputs[step]};
    else
      self.combineOutputs[step]=self.patchOutputs[step];
    end

    -- PRODUCING RNNOUTPUTS[STEP]
    --if torch.type(self.rnn) == 'nn.Recurrent' then
      --if self.train then

    if (not self.train) and self._avgAtTestFlag then
      assert(not self.efficientEval) -- avgAtTestFlag is incompatible with not maintaining stepwise outputs
      self.rnn.step=1;
      self.rnnOutputs[step]=self.rnn:updateOutput(self.combineOutputs[step]):clone()
      rnn_output_computed = true;
    else
      self.rnnOutputs[step]=self.rnn:updateOutput(self.combineOutputs[step])
    end
    -- PRODUCING RECONSTRUCTOUTPUTS[STEP]
    self.reconstructOutputs[step] = self.reconstruct:updateOutput(self.rnnOutputs[step]); -- Nx(num_rows*H)x(num_cols*W)

    -- PRODUCING ROTATED_RECONSTRUCTOUTPUTS[STEP]
    if self.rotationCompensationFlag then
      if step==1 then
        local tmp = self.selPositions[1]:clone();
        tmp:select(2,1):mul(self._viewshape[1]);
        tmp:select(2,2):mul(self._viewshape[2]);
        if not self.compensateKnownPosFlag then
          -- in this case, the network is expected to output views correctly to those position coordinates which are supplied. So we must shift along coordinates that are known.
          if self.knownElev then
            tmp:select(2,1):mul(0);
          end
          if self.knownAzim then
            tmp:select(2,2):mul(0);
          end
        end
        local zerocol = tmp:select(2,1):clone():mul(0); -- zero column corresponding to shifting of channels
        self.shiftVec=zerocol:cat(tmp,2);
      end
      self.rotated_reconstructOutputs[step] = self.rotation:updateOutput{self.reconstructOutputs[step], self.shiftVec};
    else
      self.rotated_reconstructOutputs[step] = self.reconstructOutputs[step];
    end

    if self.memorizeObservedViews then
      if torch.type(self.average_viewgrid)~= torch.type(input) then
        self.average_viewgrid = self.average_viewgrid:typeAs(input)
      end
      local h = self._glimpseshape[1]
      local w = self._glimpseshape[2]
      for t = 1,step do -- copy all previously observed views exactly on to reconstructed view grid
        for sampleIdx=1,batch_sz do -- each sample in batch
          local view = self.selPatch[t][sampleIdx]:clone();
          local viewpos = self.selPositions[t][sampleIdx];
          local y = viewpos[1]*h+1
          local x = viewpos[2]*w+1
          if self.mean_subtract_input ~= self.mean_subtract_output then
            local avg = self.average_viewgrid:narrow(2, y, h):narrow(3, x, w)
            if self.mean_subtract_input and not self.mean_subtract_output then
              -- add mean to observed view before replacing in output
              view = view + avg
            elseif self.mean_subtract_output and not self.mean_subtract_input then
             -- subtract mean from observed view before replacing in output
             view = view - avg
            end
          end
          self.rotated_reconstructOutputs[step][sampleIdx]:narrow(2, y, h):narrow(3, x, w):copy(view);
        end
      end
    end

    -- PRODUCING VECTORIZED_RECONSTRUCTOUTPUTS[STEP]
    self.vectorized_reconstructOutputs[step] = self.vectorize:updateOutput(self.rotated_reconstructOutputs[step]);

    self.output[step] = self.vectorized_reconstructOutputs[step]
    if step>1 then
      if self.train then
        mode='train';
      else
        mode='eval';
      end
      if (not self.train) and self._zeroStepsFlag and self._avgAtTestFlag then
        local errflag = torch.all(self.output[step][1]:eq(self.output[step-1][1])) --test of parameter sharing through the whole chain, across timesteps
        assert(errflag)
      end
    end
  end
  return self.output
end

function ActiveMod:updateGradInput(input, gradOutput)  -- backward pass
  local batch_sz=input:size(1);
  assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
  assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
  assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")

  -- backward through the action module
  local gradAction={}
  local gradInput_location={};
  local gradInput_patch={};
  local gradInput_input={}
  for step=self.nStep,1,-1 do
    if not (self.finetuneFactor == 0 and self.rnn_lr_mult ==0 and self.action_lr_mult == 0) then -- can skip this if freezing actor, aggregator, and the sensing apparatus below
      if self._use_action_module[step] then
        --backward through action
        assert(not self.randomActionsFlag)
        if step == 1 then
          assert(not self._zeroFirstStepFlag);
          local a=self.action:updateGradInput(self._initActionInput, self.action.output) -- no previous time step to propagate gradients to hidden layer
          assert(not torch.any(a:ne(a)), "action grad input nan");
        else
          local a=self.action.sharedClones[step]:findModules('nn.ReinforceCategorical')[1].reward
          assert(a and not torch.any(a:ne(a)), 'reward is nil or nan');
          gradAction[step]= self.action:updateGradInput(self.actionInputs[step], self.action.output)
          if self.actOnPixels then
            gradAction[step] = gradAction[step][1] -- gradient with respect to pixels not used, so getting rid of it
          end
          local a=gradAction[step];
          assert(not torch.any(a:ne(a)), 'gradAction is nan');
        end
      else
        self.action.updateGradInputStep = self.action.updateGradInputStep or self.action.step
        self.action.updateGradInputStep = self.action.updateGradInputStep and (self.action.updateGradInputStep - 1) -- just to avoid multi-step reinforce-related complications
        -- no need to set gradAction
      end
    end

    -- backward through the decoder (vectorize, rotation, reconstructor), and rnn modules
    if not (self.finetuneFactor == 0 and self.rnn_lr_mult ==0 and self.decoder_lr_mult==0) then -- can skip this if freezing reconstructor, aggregator, and the sensing apparatus below
      local gradOutput_ = gradOutput[step];
      local a=gradOutput_;
      assert(not torch.any(a:ne(a)), "gradOutput_ActiveMod nan");

      self.gradInput_vectorize[step] = self.vectorize:updateGradInput(self.rotated_reconstructOutputs[step], gradOutput_);

      if self.memorizeObservedViews then --hand-implemented backward pass
        local h = self._glimpseshape[1]
        local w = self._glimpseshape[2]
        for t = 1,step do -- copy all previously observed views exactly on to reconstructed view grid
          for sampleIdx=1,batch_sz do -- each sample in batch
            local viewpos = self.selPositions[t][sampleIdx];
            local y = viewpos[1]*h+1
            local x = viewpos[2]*w+1
            -- TODO in theory, I should be adding these gradients to the input gradients for ActiveMod, but since those are useless in my current setup, not implementing this.
            -- zero out gradients corresponding to all previously observed views
            self.gradInput_vectorize[step][sampleIdx]:narrow(2,y,h):narrow(3,x,w):zero();
          end
        end
      end
      local a=self.gradInput_vectorize[step];
      assert(not torch.any(a:ne(a)), "gradInput_vectorize nan");

      if self.rotationCompensationFlag then
        local tmp = self.rotation:updateGradInput({self.reconstructOutputs[step], self.shiftVec}, self.gradInput_vectorize[step]);
        self.gradInput_rotation[step] = tmp[1]; -- no need to pass gradients to the shift vector (comes straight from actor module, and actor module does not accept incoming gradients)
        -- TODO fix to make this cleaner and more general
      else
        self.gradInput_rotation[step] = self.gradInput_vectorize[step];
      end
      local a=self.gradInput_rotation[step];
      assert(not torch.any(a:ne(a)), "gradInput_rotation nan");

      self.gradInput_reconstruct[step] = self.reconstruct:updateGradInput(self.rnnOutputs[step], self.gradInput_rotation[step]);
      local a=self.gradInput_reconstruct[step];
      assert(not torch.any(a:ne(a)), "gradInput_reconstruct nan");
    end

    if not (self.finetuneFactor == 0 and self.rnn_lr_mult ==0) then -- can skip this if freezing aggregator and the sensing apparatus below
      -- computing gradOutput_rnn
      local gradOutput_= self.gradInput_reconstruct[step]
      local gradOutput_rnn
      if torch.type(gradOutput_)=='table' then
        gradOutput_rnn=gradOutput_[1];
      else
        gradOutput_rnn=gradOutput_
      end

      local a=gradOutput_rnn;
      assert(not torch.any(a:ne(a)), "gradOutput_rnn nan");


      -- add appropriate gradients from action module
      if step<self.nStep and not (self._iter_count<=self.pretrainModeIters) and self._use_action_module[step+1] then
        local tmp=gradAction[step+1]:clone();
        if self.knownAzim==true and self.actOnAzim==true then
          tmp=tmp:sub(1,-1,1,-2);
        end
        if self.knownElev==true and self.actOnElev==true then
          tmp=tmp:sub(1,-1,1,-2);
        end
        if self.actOnTime then
          tmp=tmp:sub(1,-1,1,-2);
        end
        tmp=tmp:sub(1,-1,1,-3) -- remove gradients w.r.t. relative locations too
        gradOutput_rnn:add(tmp);
      end
      local a=gradOutput_rnn;
      assert(not torch.any(a:ne(a)), "gradOutput_rnn nan");


      if (self._iter_count<=self.pretrainModeIters) then
        gradOutput_rnn:zero();
      end

      -- backprop
      if torch.type(self.rnn) == "nn.Recurrent" then
        --self.rnn.step = step + 1
        self.rnn:updateGradInput(self.combineOutputs[step], gradOutput_rnn) -- saves gradOutputs (sum of classification and reinforcement loss gradients) at each time-step to internal variables
      elseif torch.type(self.rnn) == "nn.LSTM" then
        --self.rnn.step = step + 1
        self.rnn:updateGradInput(self.combineOutputs[step], gradOutput_rnn) -- saves gradOutputs (sum of classification and reinforcement loss gradients) at each time-step to internal variables
      else
        abort()
      end
    end

    gradInput_input[step]=input.new():resize(batch_sz,input:size(2),input:size(3),input:size(4)):zero();
    if not (self.finetuneFactor == 0) then -- when this is zero, can save computation by not doing the backward pass here
      --print("finetuneFactor:" .. self.finetuneFactor)
      if not self.ignoreLocationFlag then
        local a= self.rnn.gradInputs[step]
        assert(not torch.any(a:ne(a)), "rnn gradInput nan");

        -- backward through the combine network
        self.gradInput_combine[step]=self.combine:updateGradInput({self.locationOutputs[step], self.patchOutputs[step]}, self.rnn.gradInputs[step]);
        local a= self.gradInput_combine[step]
        assert(not torch.any(a[1]:ne(a[1])), "gradInput_combine_entry1 nan");
        assert(not torch.any(a[2]:ne(a[2])), "gradInput_combine_entry2 nan");
      else
        self.gradInput_combine[step] = self.rnn.gradInputs[step]
      end

      if not self.ignoreLocationFlag then
        -- backward through the location network
        gradInput_location[step]=self.location:updateGradInput(self.locationInputs[step], self.gradInput_combine[step][1]);
        local a=gradInput_location[step]
        assert(not torch.any(a:ne(a)), "gradInput_location nan");
      end

      -- backward through the patch network
      gradInput_patch[step]=self.patch:updateGradInput(self.selPatch[step], self.ignoreLocationFlag and self.gradInput_combine[step] or self.gradInput_combine[step][2]);
      local a=gradInput_patch[step]
      assert(not torch.any(a:ne(a)), "gradInput_patch nan");

      for sampleIdx=1,batch_sz do
        local center=gradInput_input[step][sampleIdx]:narrow(2,self.selPositions[step][sampleIdx][1]*self._glimpseshape[1]+1, self._glimpseshape[1]):narrow(3, self.selPositions[step][sampleIdx][2]*self._glimpseshape[2]+1, self._glimpseshape[2])
        center:copy(gradInput_patch[step][sampleIdx]);
      end
    else
      --print('Skipping backward pass for frozen lower modules')
    end
    local gradInput = gradInput_input[step]
    if step == self.nStep then
      self.gradInput:resizeAs(gradInput):copy(gradInput)
    else
      self.gradInput:add(gradInput)
    end
    local a=gradInput_input[step]
    assert(not torch.any(a:ne(a)), "gradInput_input nan");
  end

  local a=self.gradInput;
  assert(not torch.any(a:ne(a)), "ActiveMod gradInput nan");

  if self.finetuneFactor~=0 then
    -- check that all modules have called updateGradInput the correct number of times
    -- will not hold for some modules when finetuneFactor is 0
    if self.IncludeActionModule and table.containsValue(self._use_action_module, true) then
      assert(self.modules[1].updateGradInputStep == 1);
    end
    local start_mod = self.IncludeActionModule and 2 or 1
    for i = start_mod,#self.modules do
      assert(self.modules[i].updateGradInputStep == 1);
    end
  end
  return self.gradInput
end

function ActiveMod:accGradParameters(input, gradOutput, scale)
  assert(self.rnn.step - 1 == self.nStep, "inconsistent rnn steps")
  assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
  assert(#gradOutput == self.nStep, "gradOutput should have nStep elements")

  for step=self.nStep,1,-1 do
    --if true then
    -- backward through the action layers
    if self._use_action_module[step] and self.action_lr_mult~=0 then
      if step == 1 then
        -- backward through initial starting actions
        self.action:accGradParameters(self._initActionInput, self.action.output, scale*self.action_lr_mult)
      else
        -- Note : gradOutput is ignored by REINFORCE modules so we give action.output as a dummy variable
        self.action:accGradParameters(self.actionInputs[step], self.action.output, scale*self.action_lr_mult)
      end

      if (self._iter_count<=self.pretrainModeIters) then
        self.action:zeroGradParameters();
      end
    else
      self.action.accGradParametersStep = self.action.accGradParametersStep or self.step
      self.action.accGradParametersStep = self.action.accGradParametersStep and (self.action.accGradParametersStep - 1) -- to avoid problems with multi-timestep reinforce()
    end

    if self.finetuneFactor ~=0 then -- if finetuneFactor is 0, can save time by not computing these
      local gradOutput_ = gradOutput[step];
      self.vectorize:accGradParameters(self.rotated_reconstructOutputs[step], gradOutput_, scale*self.decoder_lr_mult);
      if self.rotationCompensationFlag then
        rotation = self.rotation:accGradParameters({self.reconstructOutputs[step], self.shiftVec}, self.gradInput_vectorize[step], scale*self.decoder_lr_mult);
      end
      self.reconstruct:accGradParameters(self.rnnOutputs[step], self.gradInput_rotation[step], scale*self.decoder_lr_mult);
    end

    -- backward through the rnn layer
    if self.rnn_lr_mult ~=0 then
      if torch.type(self.rnn) == "nn.Recurrent" then
        --self.rnn.step = step + 1
        self.rnn:accGradParameters(self.combineOutputs[step], self.rnn.gradOutputs[step], scale*self.rnn_lr_mult)
      elseif torch.type(self.rnn) == "nn.LSTM" then
        --self.rnn.step = step + 1
        self.rnn:accGradParameters(self.combineOutputs[step], self.rnn.gradOutputs[step], scale*self.rnn_lr_mult)
      else
        abort()
      end
    end

    if self.finetuneFactor ~=0 then -- if finetuneFactor is 0, can save time by not computing these
      if not self.ignoreLocationFlag then
        -- back-propagate through combine network
          -- backward through the combine (combine) network
          self.combine:accGradParameters({self.locationOutputs[step], self.patchOutputs[step]}, self.rnn.gradInputs[step], scale*self.finetuneFactor);
      end

      -- back-propagate through location and patch networks
      if not self.ignoreLocationFlag then
        -- backward through the location network
        self.location:accGradParameters(self.locationInputs[step], self.gradInput_combine[step][1], scale*self.finetuneFactor);
      end
      -- backward through the patch network
      self.patch:accGradParameters(self.selPatch[step], self.ignoreLocationFlag and self.gradInput_combine[step] or self.gradInput_combine[step][2], scale*self.finetuneFactor);
    end
  end
  if self.finetuneFactor~=0 then
    -- check that all modules have called accGradParameters the correct number of times
    -- will not hold for some modules when finetuneFactor is 0
    if self.IncludeActionModule and table.containsValue(self._use_action_module, true) then
      assert(self.modules[1].accGradParametersStep== 1);
    end
    local start_mod = self.IncludeActionModule and 2 or 1
    for i = start_mod,#self.modules do
      assert(self.modules[i].accGradParametersStep== 1);
    end
  end
end

function ActiveMod:evaluate()
  if self.efficientEval then
    -- will forget intermediate outputs, so if you plan to do anything with stepwise outputs, then don't set this flag.
    self:applyToModules(function(module) module:evaluate() end)
  else
    self:applyToModules(function(module) module:evaluate() end)
    for i = 1,#self.modules do
      self.modules[i].train=true -- To save step-wise outputs
    end
  end
  -- parent.evaluate(self)
  self.train=false
end

function ActiveMod:training()
  self:applyToModules(function(module) module:training() end)
  -- parent.training(self)
  self.train=true
end

function ActiveMod:accUpdateGradParameters(input, gradOutput, lr)
  abort()
end

function ActiveMod:type(type)
  self._input = nil
  self._actions = nil
  self._crop = nil
  self._pad = nil
  self._byte = nil
  return parent.type(self, type)
end

function ActiveMod:__tostring__()
  local tab = '  '
  local line = '\n'
  local ext = '  |    '
  local extlast = '       '
  local last = '   ... -> '
  local str = torch.type(self)
  str = str .. ' {'
  str = str .. line .. tab .. 'action   : ' .. tostring(self.action):gsub(line, line .. tab .. ext)
  str = str .. line .. tab .. 'patch : ' .. tostring(self.patch):gsub(line, line .. tab .. ext)
  if not self.ignoreLocationFlag then
    str = str .. line .. tab .. 'location : ' .. tostring(self.location):gsub(line, line .. tab .. ext)
    str = str .. line .. tab .. 'combine : ' .. tostring(self.combine):gsub(line, line .. tab .. ext)
  end
  str = str .. line .. tab .. 'rnn      : ' .. tostring(self.rnn):gsub(line, line .. tab .. ext)
  str = str .. line .. tab .. 'reconstruct : ' .. tostring(self.reconstruct):gsub(line, line .. tab .. ext)
  str = str .. line .. tab .. 'rotation : ' .. tostring(self.rotation):gsub(line, line .. tab .. ext)
  str = str .. line .. tab .. 'vectorize : ' .. tostring(self.vectorize):gsub(line, line .. tab .. ext)
  str = str .. line .. '}'
  return str
end
