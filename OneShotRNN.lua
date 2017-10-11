local OneShotRNN, parent = torch.class("nn.OneShotRNN", "nn.Container")
require 'rnn'
-- provides a wrapper so that the different elements of an RNN are all dealt with in one call.

function OneShotRNN:__init(rnn)
  self.rnn=rnn;
  assert(torch.type(rnn)=='nn.Recurrent');
  self.rnn.copyInputs=true;
  self.modules = {self.rnn};
  self.rnn.train = true;
end

function OneShotRNN:updateOutput(input)
  self.rnn:training();
  assert(torch.type(input)=='table'); -- one element for each time step
  self.output={}
  self.rnn:forget();
  self.nStep = #input;
  for i=1,self.nStep do
      -- temporary, for testing
      -- input[i]=input[i]*0+1;
    self.output[i]=self.rnn:updateOutput(input[i]);
  end
  -- temporary.. only intended for one particular use case
  -- assert(torch.all(torch.eq(input[1], self.output[1])), "rnn modifies input");
  return self.output
end

function OneShotRNN:updateGradInput(input, gradOutput)
  assert(self.rnn.step-1 == self.nStep);
  assert(torch.type(gradOutput)=='table');
  assert(#gradOutput==self.nStep);
  self.gradInput={};
  for step=1,self.nStep do
    self.rnn.step = step+1;
    self.rnn:updateGradInput(input[step], gradOutput[step])
  end
  self.rnn:updateGradInputThroughTime()
  return self.rnn.gradInputs
end

function OneShotRNN:accGradParameters(input, gradOutput, scale)
  assert(self.rnn.step-1 == self.nStep);
  assert(torch.type(gradOutput)=='table');
  assert(#gradOutput==self.nStep);
  for step=1,self.nStep do
    self.rnn.step = step + 1
    self.rnn:accGradParameters(gradOutput[step], self.rnn.gradOutputs[step], scale)
  end
  -- back-propagate through time (BPTT)
  self.rnn:accGradParametersThroughTime()
end
