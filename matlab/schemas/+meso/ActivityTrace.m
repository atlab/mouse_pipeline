%{
# deconvolved calcium acitivity
-> meso.ScanSetUnit
-> shared.SpikeMethod
---
-> meso.Activity
trace                       : longblob                      # 
%}


classdef ActivityTrace < dj.Computed

	methods(Access=protected)

		function makeTuples(self, key)
		%!!! compute missing fields for key here
% 			 self.insert(key)
		end
	end

end