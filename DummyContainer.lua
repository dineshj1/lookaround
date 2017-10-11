------------------------------------------------------------------------
--[[ DummyContainer ]]--
-- for easy debugging...
-- e.g. if you want to stop at a particular layer in the forward pass, put it inside a dummycontainer, and set the breakpoint inside DummyContainer:updateOutput()
------------------------------------------------------------------------
local DummyContainer, parent = torch.class("nn.DummyContainer", "nn.Container")

function DummyContainer:__init(module)
  self.modules={module}
end

function DummyContainer:updateOutput(input) -- forward computation
  --print("Entering DummyContainer (forward)")
  --require('mobdebug').start()
  self.output=self.modules[1]:updateOutput(input);
  --print(#self.output)
  --print("Exiting DummyContainer (forward)")
  return self.output
end

function DummyContainer:updateGradInput(input, gradOutput)  -- backward pass
  self.gradInput = self.modules[1]:updateGradInput(input, gradOutput);
  return self.gradInput
end

function DummyContainer:accGradParameters(input, gradOutput, scale)
  self.modules[1]:accGradParameters(input,gradOutput,scale);
end

function DummyContainer:backwardThroughTime()
end

function DummyContainer:training()
  self.modules[1]:training();
  parent.training(self)
  assert(self.train == true)
end

function DummyContainer:evaluate()
  self.modules[1]:evaluate()
  parent.evaluate(self)
  assert(self.train == false)
end

function DummyContainer:reinforce(reward)
  local modules = self.modules
  self.modules = nil
  local ret = parent.reinforce(self, reward)
  self.modules = modules
  return ret
end

function DummyContainer:type(type)
  return nn.Sequencer.type(self, type)
end

function DummyContainer:__tostring__()
  local tab = '  '
  local line = '\n'
  local ext = '  |    '
  local last = '   ... -> '
  local str = torch.type(self)
  str = str .. ' {'
  str = str .. line .. tab .. 'module: ' .. tostring(self.modules[1]):gsub(line, line .. tab .. ext)
  str = str .. line .. '}'
  return str
end
