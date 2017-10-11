------------------------------------------------------------------------
--[[ MSEFeedback ]]--
-- Feedback
------------------------------------------------------------------------
local MSEFeedback, parent = torch.class("dp.MSEFeedback", "dp.Feedback")

function MSEFeedback:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1],
      "Constructor requires key-value arguments")
   local args, name, target_dim, output_module, target_shape = xlua.unpack(
      {config},
      'MSEFeedback',
      '',
      {arg='name', type='string', default='MSE',
       help='name identifying Feedback in reports'},
      {arg='target_dim', type='number', default=-1,
       help='index of target vector to measure MSE against'},
      {arg='output_module', type='nn.Module',
       help='module applied to output before measuring mean squared error'},
      {arg='target_shape', type=string,
       help='shape of batch targets'}
   )
   config.name = name
   self._output_module = output_module or nn.Identity()
   self._target_dim=target_dim;
   parent.__init(self, config)
   self._target_shape=target_shape or 'cbw';
end

function MSEFeedback:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function MSEFeedback:doneEpoch(report)
   if self.mse and self._verbose then
      print(self._id:toString().." mse = "..self.mse)
   end
end

function MSEFeedback:_add(batch, output, report)
   if self._output_module then
      output = self._output_module:updateOutput(output)
   end
   local tgt = batch:targets():forward(self._target_shape)
   if self._target_dim >0 then
      tgt=tgt[self._target_dim]
   end
   if torch.type(tgt)~=torch.type(output) then
      tgt=tgt:typeAs(output);
   end

   local diff=output:clone();
   diff:csub(tgt); -- (input-target)
   local err=diff:norm(2,2):pow(2)/diff:size(2);

   self.sse=self.sse+err:sum();
   self.count=self.count+err:size(1);
   self.mse=self.sse/self.count;
end

function MSEFeedback:_reset()
   self.sse=0
   self.count=0
   self.mse=nil
end

function MSEFeedback:report()
   return {
      [self:name()] = {
         mse = self.mse;
         },
      n_sample = self.count
   }
end
