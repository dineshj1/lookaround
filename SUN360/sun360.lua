------------------------------------------------------------------------
--[[ SUN360 ]]--
------------------------------------------------------------------------
require 'dp'
require 'config'

local SUN360, DataSource = torch.class("dp.SUN360", "dp.DataSource")
SUN360.isSUN360 = true

SUN360._name = 'sun360'
SUN360._image_axes = 'bcwh'
SUN360._pano_axes = 'bcwh'

function SUN360:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1],
      "Constructor requires key-value arguments")
   local args, load_all, input_preprocess, target_preprocess
   args, self._shuffleseed,
         load_all,
         input_preprocess,
         target_preprocess,
         self._silent,
         self._include_cls_labels,
         self._include_saliency_scores,
         self._mean_subtract_input,
         self._mean_subtract_output,
         self._only_test,
         self._noTest,
         self._mini
      = xlua.unpack(
      {config},
      'SUN360',
      'Panorama reconstruction problem.',
      {arg='shuffleseed', type=number, default=62346,
       help='random seed for shuffling and splitting'},
      {arg='load_all', type='boolean',
       help='Load all datasets : train, valid, test.', default=true},
      {arg='input_preprocess', type='table | dp.Preprocess',
       help='to be performed on set inputs, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'},
      {arg='target_preprocess', type='table | dp.Preprocess',
       help='to be performed on set targets, measuring statistics ' ..
       '(fitting) on the train_set only, and reusing these to ' ..
       'preprocess the valid_set and test_set.'},
      {arg='silent', type=number, default=true, help='whether or not to print updates'},
      {arg='include_cls_labels', type=number, default=true},
      {arg='include_saliency_scores', type=boolean, default=false, help='/gbvs_saliency_scores loaded (for test set alone)'},
      {arg='mean_subtract_input', type=boolean, default=false, help='whether to subtract mean from pixels fed into network'},
      {arg='mean_subtract_output', type=boolean, default=false, help='whether to subtract mean from pixels output from network'},
      {arg='only_test', type=boolean, default=false, help='load only test set'},
      {arg='noTest', type=boolean, default=false, help='do not load test set'},
      {arg='mini', type=boolean, default=false, help='whether or not to use mini feature files (for debugging)'}
   )
   if not self._mini then
     self._train_hdf5_file      = [config_ipdir 'torchfeed/pixels_trn_torchfeed.h5';
     self._val_hdf5_file        = [config_ipdir 'torchfeed/pixels_val_torchfeed.h5';
     self._test_hdf5_file       = [config_ipdir 'torchfeed/pixels_tst_torchfeed.h5';
   else
     self._train_hdf5_file      = [config_ipdir 'minitorchfeed/pixels_trn_torchfeed.h5';
     self._val_hdf5_file        = [config_ipdir 'minitorchfeed/pixels_val_torchfeed.h5';
     self._test_hdf5_file       = [config_ipdir 'minitorchfeed/pixels_tst_torchfeed.h5';
   end

   if load_all then
      self:loadAll()
    elseif self._only_test then
      self:loadTest()
   end
   DataSource.__init(self, {
      train_set=self:trainSet(), valid_set=self:validSet(),
      test_set=self:testSet(), input_preprocess=input_preprocess,
      target_preprocess=target_preprocess
   })
end

function SUN360:loadTrain()
   require 'hdf5'
   print('reading ' .. self._train_hdf5_file);
   local myfile = hdf5.open(self._train_hdf5_file);
   local trn_data={};
   trn_data[2]=myfile:read('/labs'):all():squeeze();
   trn_data[3]=myfile:read('/target_viewgrid'):all():squeeze():float()/255;
   trn_data[1]=trn_data[3]:clone();
   if self._mean_subtract_input then
     self._avg_viewgrid = self._avg_viewgrid or myfile:read('/average_target_viewgrid'):all():float()/255;
     local tmp = torch.repeatTensor(self._avg_viewgrid, trn_data[1]:size(1), 1, 1)
     trn_data[1] = trn_data[1] - tmp;
   else
     print('Warning: not mean-subtracting from input.');
   end
   if self._mean_subtract_output then
     self._avg_viewgrid = self._avg_viewgrid or myfile:read('/average_target_viewgrid'):all():float()/255;
     local tmp = torch.repeatTensor(self._avg_viewgrid, trn_data[3]:size(1), 1, 1)
     trn_data[3] = trn_data[3] - tmp;
   end
   if self._include_cls_labels then
     self.trn_labs=trn_data[2];
   end
   self.gridshape=myfile:read('/gridshape'):all():type('torch.IntTensor');
   self.feat_snapshape=myfile:read('/view_snapshape'):all():type('torch.IntTensor');
   self.view_snapshape=myfile:read('/view_snapshape'):all():type('torch.IntTensor');
   myfile:close();
   print('done');

   self._image_size={trn_data[1]:size(2), trn_data[1]:size(3), trn_data[1]:size(4)};
   self._montage_size={trn_data[3]:size(2), trn_data[3]:size(3), trn_data[3]:size(4)};
   self._image_channels=trn_data[1]:size(2);

   self._num_trn_imgs=trn_data[1]:size(1);
   --local indices = torch.randperm(self._num_trn_imgs):long()
   sel_samples = trn_data[1]--:index(1, indices) -- NxC*W*H
   sel_targets = trn_data[3]--:index(1, indices) -- NxC*H*W

   if not self._silent then
     print("Creating train data");
   end

   self:trainSet(
     self:createDataSet(
         sel_samples, sel_targets:view(1,sel_targets:size(1),-1),
         'train'
     )
   )

   return self:trainSet()
end


function SUN360:loadVal()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.

   require 'hdf5'
   print('reading ' .. self._val_hdf5_file);
   local myfile = hdf5.open(self._val_hdf5_file);
   local val_data={};
   val_data[2]=myfile:read('/labs'):all():squeeze();
   val_data[3]=myfile:read('/target_viewgrid'):all():squeeze():float()/255;
   val_data[1]=val_data[3]:clone();
   if self._mean_subtract_input then
     self._avg_viewgrid = self._avg_viewgrid or myfile:read('/average_target_viewgrid'):all():float()/255;
     local tmp = torch.repeatTensor(self._avg_viewgrid, val_data[1]:size(1), 1, 1)
     val_data[1] = val_data[1] - tmp;
   else
     print('Warning: not mean-subtracting from input.');
   end
   if self._mean_subtract_output then
     self._avg_viewgrid = self._avg_viewgrid or myfile:read('/average_target_viewgrid'):all():float()/255;
     local tmp = torch.repeatTensor(self._avg_viewgrid, val_data[3]:size(1), 1, 1)
     val_data[3] = val_data[3] - tmp;
   end
   if self._include_cls_labels then
     self.val_labs=val_data[2];
   end

   if not self.gridshape then
     self.gridshape=myfile:read('/gridshape'):all():type('torch.IntTensor');
     self.feat_snapshape=myfile:read('/view_snapshape'):all():type('torch.IntTensor');
     self.view_snapshape=myfile:read('/view_snapshape'):all():type('torch.IntTensor');
   end
   myfile:close();
   print('done');

   self:validSet(
     self:createDataSet(
       val_data[1], val_data[3]:view(1,val_data[3]:size(1),-1),
       'valid'
     )
   )
   return self:validSet()
end
function SUN360:loadTest()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.

   require 'hdf5'
   print('reading ' .. self._test_hdf5_file);
   local myfile = hdf5.open(self._test_hdf5_file);
   local test_data={};
   test_data[2]=myfile:read('/labs'):all():squeeze();
   test_data[3]=myfile:read('/target_viewgrid'):all():squeeze():float()/255;
   test_data[1]=test_data[3]:clone();
   if self._mean_subtract_input then
     self._avg_viewgrid = self._avg_viewgrid or myfile:read('/average_target_viewgrid'):all():float()/255;
     local tmp = torch.repeatTensor(self._avg_viewgrid, test_data[1]:size(1), 1, 1)
     test_data[1] = test_data[1] - tmp;
   else
     print('Warning: not mean-subtracting from input.');
   end
   if self._mean_subtract_output then
     self._avg_viewgrid = self._avg_viewgrid or myfile:read('/average_target_viewgrid'):all():float()/255;
     local tmp = torch.repeatTensor(self._avg_viewgrid, test_data[3]:size(1), 1, 1)
     test_data[3] = test_data[3] - tmp;
   end
   if self._include_cls_labels then
     self.tst_labs=test_data[2];
   end
   if self._include_saliency_scores then
     print("Loading saliency scores")
     self.tst_sal_scores=myfile:read('/gbvs_saliency_scores'):all():transpose(2,3);
   end

   if not self.gridshape then
     self.gridshape=myfile:read('/gridshape'):all():type('torch.IntTensor');
     self.feat_snapshape=myfile:read('/view_snapshape'):all():type('torch.IntTensor');
     self.view_snapshape=myfile:read('/view_snapshape'):all():type('torch.IntTensor');
   end
   myfile:close();
   print('done');

     self:testSet(
     self:createDataSet(
         test_data[1], test_data[3]:view(1,test_data[3]:size(1),-1),
         'test'
     )
   )
   return self:testSet()
end
function SUN360:loadAll()
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   self:loadTrain()
   self:loadVal()
   self:loadTest()
   return self:trainSet(), self:validSet(), self:testSet()
end

--Creates a SUN360 Dataset out of inputs, targets and which_set
function SUN360:createDataSet(inputs, targets, which_set)
   local input_v = dp.ImageView()
   input_v:forward(self._image_axes, inputs)
   local target_v = dp.SequenceView()
   target_v:forward('cbw', targets)
   local ds = dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
   ds:ioShapes('bcwh', 'cbw')
   return ds
end
