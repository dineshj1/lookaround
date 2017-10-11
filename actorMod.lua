-----------------------------------------------------------------------
--[[ ActorMod ]]--
-- Ref. A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf
-- module which takes an RNN as argument with other
-- hyper-parameters such as the maximum number of steps,
-- action (actions sampling module like ReinforceNormal) and
------------------------------------------------------------------------
local ActorMod, parent = torch.class("nn.ActorMod", "nn.Container")

function ActorMod:__init(config)
  config = config or {}
  assert(torch.type(config) == 'table' and not config[1], "Constructor requires key-value arguments")
  local args
  args,
  self.presample_mod,
  self.sigmoid_mod,
  self.reinforce_mod,
  self.postsample_mod,
  self.straightThroughFlag = xlua.unpack(
     {config},
     'ActorMod',
     '',
     {arg='presample_mod', type='nn.Module'},
     {arg='sigmoid_mod', type='nn.Module', 'should have same input and output dimensions'},
     {arg='reinforce_mod', type='nn.Module', 'should be a REINFORCE module'},
     {arg='postsample_mod', type='nn.Module', 'should haven no parameters'},
     {arg='straightThroughFlag', default=false, type='boolean'}
  )

  if not self.straightThroughFlag then
    self.full_mod = nn.Sequential():add(self.presample_mod):add(self.sigmoid_mod):add(self.reinforce_mod):add(self.postsample_mod);
    self.modules = {self.full_mod}
  else
    -- no full_mod
    self.modules={self.presample_mod, self.sigmoid_mod, self.reinforce_mod, self.postsample_mod}
  end
end

function ActorMod:updateOutput(input) -- forward computation
  if not self.straightThroughFlag then
    self.output = self.full_mod:updateOutput(input);
  else
    self.presample_output = self.presample_mod:updateOutput(input);
    self.sigmoid_output = self.sigmoid_mod:updateOutput(self.presample_output);
    self.reinforce_output = self.reinforce_mod:updateOutput(self.sigmoid_output);
    self.output= self.postsample_mod:updateOutput(self.reinforce_output);
  end

  return self.output
end

function ActorMod:updateGradInput(input, gradOutput)  -- backward pass
  if not self.straightThroughFlag then
    self.gradInput = self.full_mod:updateGradInput(input, self.output); -- gradOutput is ignored, since this is a stochastic network
  else
    self.gradInput_reinforce = self.reinforce_mod:updateGradInput(
        self.sigmoid_output, self.reinforce_output)
    -- skip the sigmoid!
    self.gradInput = self.presample_mod:updateGradInput(
        input, self.gradInput_reinforce);
  end
  return self.gradInput
end

function ActorMod:accGradParameters(input, gradOutput, scale)
  if not self.straightThroughFlag then
    self.full_mod:accGradParameters(input, self.output); -- gradOutput is ignored, since this is a stochastic network
  else
    self.reinforce_mod:updateGradInput(
        self.sigmoid_output, self.reinforce_output)
    -- skip the sigmoid!
    self.presample_mod:updateGradInput(
        input, self.gradInput_reinforce);
  end
  return self.gradInput
end

function ActorMod:accUpdateGradParameters(input, gradOutput, lr)
  abort()
end

--function ActorMod:training()
--  self:applyToModules(function(module) module:training() end)
--  --parent.train(self)
--  self.train=true
--  assert(self.train==true)
--end
--
--function ActorMod:evaluate()
--  self:applyToModules(function(module) module:evaluate() end)
--  --parent.evaluate(self)
--  self.train=false
--  assert(self.train==false)
--end

--function ActorMod:reinforce(reward)
--  self:applyToModules(function(module) module:reinforce(reward) end)
--  local modules = self.modules
--  self.modules = nil
--  local ret = parent.reinforce(self, reward)
--  self.modules = modules
--  return ret
--end

function ActorMod:type(type)
  self._input = nil
  self._actions = nil
  self._crop = nil
  self._pad = nil
  self._byte = nil
  return parent.type(self, type)
end

function ActorMod:__tostring__()
  local tab = '  '
  local line = '\n'
  local ext = '  |    '
  local extlast = '       '
  local last = '   ... -> '
  local str = torch.type(self)  .. '( '
  str = str .. (self.straightThroughFlag and 'straightThrough' or 'REINFORCE')
  str = str .. ' )'
  str = str .. ' {'
  if not self.straightThroughFlag then
     str = str .. line .. tab ..   tostring(self.full_mod):gsub(line, line .. tab .. ext)
  else
    str = str .. line .. tab .. 'presample_mod: ' .. tostring(self.presample_mod):gsub(line, line .. tab .. ext)
    str = str .. line .. tab .. 'sigmoid_mod (ignored on backward pass): ' .. tostring(self.sigmoid_mod):gsub(line, line .. tab .. ext)
    str = str .. line .. tab .. 'reinforce_mod: ' .. tostring(self.reinforce_mod):gsub(line, line .. tab .. ext)
    str = str .. line .. tab .. 'postsample_mod: ' .. tostring(self.postsample_mod):gsub(line, line .. tab .. ext)
  end
  str = str .. line .. '}'
  return str
end
