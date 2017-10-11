local CircShift, parent = torch.class("nn.CircShift", "nn.Module")
-- can either set shift vector during construction, or just pass it in as the second element of a table argument.. note that no gradients will be backpropagated to this second input though (circular shift is not differentiable with respect to the shift vector)

--require 'os'

function CircShift:__init(config)
  config=config or {}
  assert(torch.type(config) == 'table' and not config[1], "Constructor requires key-value arguments")
  local args
  args, self.shiftVec = xlua.unpack(
    {config},
    {'CircShift'},
    '',
    {arg='shiftVec', default={0,0}, help='how much to shift input by along each dimension'}
  )
  parent.__init(self);
end

function CircShift:circshift_idx(shift, size)
  --print(torch.range(size-shift+1,size):cat(torch.range(1,size-shift)))
  return torch.range(size-shift+1,size):cat(torch.range(1,size-shift)):type('torch.LongTensor');
end

function CircShift:updateOutput(input)
  local inputMat, shiftVec
  if torch.type(input)=='table' then
    inputMat =input[1];
    shiftVec=input[2];
  else
    inputMat=input;
    --shiftVec=self.shiftVec;
  end
  --local start_time=os.clock();
  self.output=inputMat:clone();
  local nSample = inputMat:size(1);
  for sampleno = 1, nSample do
    local sample_shiftVec=shiftVec and shiftVec[sampleno] or self.shiftVec;
    --print(sample_shiftVec)
    for dim = 1,inputMat:dim()-1 do
      if not (sample_shiftVec[dim]==0 or sample_shiftVec[dim]==inputMat:size(dim+1)) then
        assert(sample_shiftVec[dim]>0)
        assert(sample_shiftVec[dim]<inputMat:size(dim+1))
        self.output[sampleno] = self.output[sampleno]:index(dim, self:circshift_idx(sample_shiftVec[dim], inputMat:size(dim+1)));
        --self.output[sampleno]:indexCopy(dim, self:circshift_idx(inputMat:size(dim+1)-sample_shiftVec[dim], inputMat:size(dim+1)), self.output[sampleno]); --not working
      end
    end
  end
  --local end_time=os.clock();
  --print("Elapsed time:" .. end_time-start_time .. "s");
  self.saved_shiftVec=shiftVec;
  return self.output
end

function CircShift:updateGradInput(input, gradOutput)
  local inputMat, shiftVec
  if torch.type(input)=='table' then
    inputMat=input[1];
  else
    inputMat=input;
  end
  self.gradInput=gradOutput:clone();
  shiftVec=self.saved_shiftVec;
  local nSample = gradOutput:size(1);
  for sampleno = 1, nSample do
    local sample_shiftVec=shiftVec[sampleno];
    for dim = 1,gradOutput:dim()-1 do
      if not (sample_shiftVec[dim]==0 or sample_shiftVec[dim]==inputMat:size(dim+1)) then
        assert(sample_shiftVec[dim]>0)
        assert(sample_shiftVec[dim]<inputMat:size(dim+1))
        self.gradInput[sampleno] = self.gradInput[sampleno]:index(dim, self:circshift_idx(inputMat:size(dim+1)-sample_shiftVec[dim], inputMat:size(dim+1)));
        --self.gradInput[sampleno]:indexCopy(dim, self:circshift_idx(sample_shiftVec[dim], inputMat:size(dim+1)), self.gradInput[sampleno]); --not working
      end
    end
  end
  self.gradInput = {self.gradInput, input[2]:clone():zero()}
  return self.gradInput
end
