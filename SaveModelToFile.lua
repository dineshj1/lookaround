------------------------------------------------------------------------
--[[ SaveModelToFile ]]--
-- Strategy. Not an Observer.
-- Saves version of the subject with the lowest error
------------------------------------------------------------------------
local SaveModelToFile = torch.class("dp.SaveModelToFile")
SaveModelToFile.isSaveModelToFile = true

function SaveModelToFile:__init(config)
   config = config or {}
   assert(not config[1], "Constructor requires key-value arguments")
   local args, in_memory, save_dir, verbose = xlua.unpack(
      {config},
      'SaveModelToFile',
      'Saves version of the subject with the lowest error',
      {arg='in_memory', type='boolean', default=false,
       help='only saves the subject to file at the end of the experiment'},
      {arg='save_dir', type='string', help='defaults to dp.SAVE_DIR'},
      {arg='verbose', type='boolean', default=true,
       help='can print messages to stdout'}
   )
   self._in_memory = in_memory
   self._save_dir = save_dir or dp.SAVE_DIR
   self._verbose = verbose
end

function SaveModelToFile:setup(subject, mediator)
   self._mediator = mediator
   if self._in_memory then
      self._mediator:subscribe('doneExperiment', self, 'doneExperiment')
   end

   --concatenate save directory with subject id
   self._filename = paths.concat(self._save_dir, subject:id():toPath() .. '.dat')
   os.execute('mkdir -p ' .. sys.dirname(self._filename))
end

function SaveModelToFile:filename()
   return self._filename
end

function SaveModelToFile:save(subject)
   assert(subject, "SaveModelToFile not setup error")
   assert(subject._model, "subject does not have member _model")
   if self._in_memory then
      dp.vprint(self._verbose, 'SaveModelToFile: serializing subject to memory')
      self._save_cache = nil
      self._save_cache = torch.serialize(subject._model)
   else
      dp.vprint(self._verbose, 'SaveModelToFile: saving to '.. self._filename)
      return torch.save(self._filename, subject._model:clone():forget():float())
   end
end

function SaveModelToFile:doneExperiment()
   if self._in_memory and self._save_cache then
      dp.vprint(self._verbose, 'SaveModelToFile: saving to '.. self._filename)
      local f = io.open(self._filename, 'w')
      f:write(self._save_cache)
      f:close()
   end
end

-- the following are called by torch.File during [un]serialization
function SaveModelToFile:write(file)
   -- prevent subject from being serialized twice
   local state = _.map(self,
      function(k,v)
         if k ~= '_save_cache' then
            return v;
         end
      end)

   file:writeObject(state)
end

function SaveModelToFile:read(file)
   local state = file:readObject()
   for k,v in pairs(state) do
      self[k] = v
   end
end

function SaveModelToFile:verbose(verbose)
   self._verbose = (verbose == nil) and true or verbose
end

function SaveModelToFile:silent()
   self:verbose(false)
end
