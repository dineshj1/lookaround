 ------------------------------------------------------------------------
--[[ IdentityFeedback ]]--
-- Feedback
------------------------------------------------------------------------
local IdentityFeedback, parent = torch.class("dp.IdentityFeedback", "dp.Feedback")

function IdentityFeedback:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1],
      "Constructor requires key-value arguments")
   local args, name, output_module = xlua.unpack(
      {config},
      'IdentityFeedback',
      '',
      {arg='name', type='string', default='identity',
       help='name identifying Feedback in reports'},
      {arg='output_module', type='nn.Module',
       help='module applied to output before measuring confusion matrix'}
   )
   config.name = name
   self._output_module = output_module or nn.Identity()
   parent.__init(self, config)
end

function IdentityFeedback:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function IdentityFeedback:doneEpoch(report)
   if self.avg and self._verbose then
      print(self._id:toString().." avgloss = "..self.avg)
   end
end

function IdentityFeedback:_add(batch, output, report)
   if self._output_module then
      output = self._output_module:updateOutput(output)
   end
   -- batch is dummy variable in this case
   if not self.sum then
     self.sum=0
     self.count=0
     self.avg=nil
   end

   self.sum=self.sum+output:sum();
   self.count=self.count+output:size(1);
   self.avg=self.sum/self.count;
end

function IdentityFeedback:_reset()
   self.sum=0
   self.count=0
   self.avg=nil
end

function IdentityFeedback:report()
   return {
      [self:name()] = {
         avg = self.avg;
         },
      n_sample = self.count
   }
end
