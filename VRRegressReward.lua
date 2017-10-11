------------------------------------------------------------------------
--[[ VRRegressReward ]]--
-- Variance reduced vector regression reinforcement criterion.
-- input : {predicted vector, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRRegressReward, nn.SelectTable(-1))
------------------------------------------------------------------------
local VRRegressReward, parent = torch.class("nn.VRRegressReward", "nn.Criterion")

function VRRegressReward:__init(config)
   parent.__init(self)

   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], "Constructor requires key-value arguments")
   local args
   args,
   self.module,
   self.perstep_scale,
   self.finstep_scale,
   self.criterion,
   self.offset,
   self.sizeAverage,
   self.discount,
   self.improvementReward,
   self.baseline_lr_factor
   = xlua.unpack(
      {config},
      'VRRegressReward',
      '',
      {arg='module', type='nn.Module'},
      {arg='perstep_scale', type='number', default = 1, 'reward scale for each non-final time instant. As discount factor goes to 1, makes sense to set perstep_scale to 0.'},
      {arg='finstep_scale', type='number', default = 1, 'reward scale for final time instant'},
      {arg='criterion', type='nn.Criterion', default = nn.MSECriterion(), 'criterion to be used in training baseline'},
      {arg='offset', type=number, default = 0, 'reward offset'},
      {arg='sizeAverage', type=boolean, default = true, 'if true, normalizes sample reward by size of batch'},
      {arg='discount', type=number, default = 1, 'if 0, then purely greedy rewards. if 1, then all future rewards. As things get more greedy i.e discount = 0, makes sense to set perstep_scale to 1.'},
      {arg='improvementReward', type=boolean, default = false, 'if true, non-final time-instant rewards are only improvements over previous time step'},
      {arg='baseline_lr_factor', type='number', default = 1, 'to allow controlling learning rate independent of baseline ...'}
   )

   if self.improvementReward then
     print("improvementReward NOT IMPLEMENTED")
     abort()
   end
end

function VRRegressReward:updateOutput(input_baseline, target)
   assert(torch.type(input_baseline) == 'table')
   local input = input_baseline[1]; -- table of reconstructions
   assert(torch.type(input) == 'table')
   self.num_steps = #input
   self.reward = {}
   self.step_output = {}
   self.output=0;
   -- TODO also implement an improvement reward option?
   for t=1,self.num_steps do
     local step_input = self:toBatch(input[t],1)
     if t==self.num_steps or self.perstep_scale>0 then
       if torch.type(step_input)~=torch.type(target) then
         target=target:typeAs(step_input);
       end

       local diff=step_input:clone();
       diff:csub(target); -- (input-target)
       local err=diff:norm(2,2):pow(2)/diff:size(2);

       -- reward is negative of err, scaled by self.scale
       self.reward[t]=self.reward[t] or step_input.new()
       self.reward[t]:resize(err:size(1)):copy(self.offset-err) --:mul(t == self.num_steps and self.finstep_scale or self.perstep_scale);

       -- loss = -sum(reward)
       self.step_output[t] = -self.reward[t]:sum()
       if self.sizeAverage then
          self.step_output[t] = self.step_output[t]/step_input:size(1)
       end
       self.output = self.output + self.step_output[t]
     else
       self.reward[t]=self.reward[t] or step_input.new():resize(step_input:size(1))
     end
   end
   return self.output
end

function VRRegressReward:updateGradInput(input_baseline, target)
   local input = input_baseline[1]
   local baseline = input_baseline[2]
   assert(torch.type(input) == 'table')
   assert(torch.type(baseline) == 'table')


   self.vrReward= self.vrReward or {}
   self.gradInput={}
   self.gradInput[1]={}
   self.gradInput[2]={}

   for t=1,self.num_steps do
     local step_input = self:toBatch(input[t],1)
     local step_baseline = self:toBatch(baseline[t],1)
     -- zero gradInput (this criterion has no gradInput for the reconstruction)
     self.gradInput[1][t] = self.gradInput[1][t] or step_input.new()
     self.gradInput[1][t]:resizeAs(step_input):zero()
     self.gradInput[1][t] = self:fromBatch(self.gradInput[1][t], 1)

     if t==self.num_steps or self.perstep_scale>0 then
       -- reduce variance of reward using baseline
       self.vrReward[t] = self.reward[t].new()
       self.vrReward[t]:resizeAs(self.reward[t]):copy(self.reward[t])
       self.vrReward[t]:add(-1, step_baseline)
       self.vrReward[t]:mul(t == self.num_steps and self.finstep_scale or self.perstep_scale);

       if self.sizeAverage then
          self.vrReward[t]:div(step_input:size(1))
       end

       -- learn the baseline reward
       self.gradInput[2][t] = self.gradInput[2][t] or step_baseline.new()
       self.criterion:forward(step_baseline, self.reward[t])
       self.gradInput[2][t] = self.criterion:backward(step_baseline, self.reward[t])
       self.gradInput[2][t]:mul(self.baseline_lr_factor);
       self.gradInput[2][t] = self:fromBatch(self.gradInput[2][t], 1)
     else
       self.vrReward[t] = self.reward[t]:clone():zero()
       self.gradInput[2][t] = self.gradInput[2][t] or step_baseline.new()
       self.gradInput[2][t]:resizeAs(step_baseline):zero()
       self.gradInput[2][t] = self:fromBatch(self.gradInput[2][t], 1)
     end
   end

   local running_sum={}
   for t1=1,self.num_steps-1 do
     running_sum[t1]=self.vrReward[t1]
     for t2=t1+1,self.num_steps do
       running_sum[t1]=running_sum[t1]+self.discount^(t2-t1)*self.vrReward[t2]
     end
   end
   self.module:reinforce(running_sum);

   return self.gradInput
end

function VRRegressReward:type(type)
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
