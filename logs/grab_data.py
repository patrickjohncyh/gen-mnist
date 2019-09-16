import torch

files = [ ('2x2-no-opt-e100m-accy-curves.pt' , '2x2-no-opt-e200m-accy-curves.pt'),
		  ('2x2-opt-e100m-accy-curves.pt'    , '2x2-opt-e150m-accy-curves.pt'),
		  ('3x3-no-opt-e100m-accy-curves.pt' , '3x3-no-opt-e200m-accy-curves.pt'),
		  ('3x3-opt-e100m-accy-curves.pt'    , '3x3-opt-e150m-accy-curves.pt'),
		  ('4x4-no-opt-e100m-accy-curves.pt' , '4x4-no-opt-e200m-accy-curves.pt'),
		  ('4x4-opt-e100m-accy-curves.pt'    , '4x4-opt-e150m-accy-curves.pt'),
		  ('5x5-no-opt-e100m-accy-curves.pt' , '5x5-no-opt-e200m-accy-curves.pt'),
		  ('6x6-no-opt-e100m-accy-curves.pt' , '6x6-no-opt-e200m-accy-curves.pt')]


for file in files:
	nodes = int(file[0][0])**2
	x = torch.load(file[0])
	y = torch.load(file[1])

	y_train,y_val = [f[0] for f in y[nodes]],[f[1] for f in y[nodes]]
	x_train,x_val = [f[0] for f in x[nodes]],[f[1] for f in x[nodes]]
	z_val = x_val+y_val
	z_train = x_train+y_train

	print(file[0])
	print(z_val)
	print(z_train)
