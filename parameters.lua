require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'
require 'dpnn'

if not opt then
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('GridNet training')
	cmd:text()
	cmd:text('Options:')
	cmd:text()
	cmd:text('Data options:')
	cmd:option('-extraRatio',0.5,'Ratio of extra (coarse) data used')
	cmd:option('-scaleMin',1, 'Minimum scaling for the cropAndScale preprocessing')
	cmd:option('-scaleMax',2, 'Maximum scaling for the cropAndScale preprocessing')
	cmd:option('-sizeX', 512, 'Input image width')
	cmd:option('-sizeY', 512, 'Input image height')
	cmd:option('-hflip', false, 'Use horizontal flip randomly')
	cmd:text()
	cmd:text('Model options:')
	cmd:option('-model','','Model file for a pretrained model or empty to train from scratch')
	cmd:option('-nColumns',3,'Number of columns for the conv part')
	cmd:option('-dropFactor',0.1,'Dropout factor for the TotalDropout operator')
	cmd:text()
	cmd:text('Gradient descent parameters :')
	cmd:option('-learningRate', 0.01, 'learning rate at t=0 (for sgd, rmsprop and adam)')
	cmd:option('-learningRateDecay', 5e-7, 'learning rate decay (for sgd and adam)')
	cmd:option('-epsilon',1e-8,'Value with which to initialise m (for rmsprop and adam)')
	cmd:option('-beta1',0.9, 'first moment coefficient (adam)')
	cmd:option('-beta2',0.999, 'second moment coefficient (adam)')
	cmd:option('-batchSize', 4, 'mini-batch size')
	cmd:option('-nbIterationTrain',800,'Number of iteration per training epoch')
	cmd:option('-nbIterationValid',200,'Number of iteration per validation epoch')
	cmd:text()
	cmd:text('Others :')
	cmd:option('-save', '', 'subdirectory to save/log experiments in')
	cmd:option('-seed', -1, 'Seed used for the random generator')
	cmd:option('-numthreads',1,'Number of threads used by torch')
	cmd:option('-silent',false,'Print nothing on the standards output')
	cmd:text()
	cmd:text('GPU Options :')
	cmd:option('-device',1, 'Wich GPU device to use')
	opt = cmd:parse(arg or {})

	--if opt.silent then
	--	cmd:silent()
	--end

	if opt.seed == -1 then
		opt.seed = torch.initialSeed()
	else
		torch.manualSeed(opt.seed)
	end


	opt.save = "results/" .. opt.save .. os.date("_%a-%d-%b-%Hh-%Mm-%Ss")
	paths.mkdir(opt.save)
	cmd:log(opt.save .. '/log.txt', opt)
	print("==> Save results into: " .. opt.save)

	print "*** Cuda activated ***"
	require 'cunn'
	require 'cudnn'

	cudnn.benchmark = true
	cudnn.fastest   = true
	cudnn.verbose   = false

	assert(opt.device <= cutorch.getDeviceCount(), "Error GPU device > #number GPU")
	cutorch.setDevice(opt.device)

	torch.setnumthreads(opt.numthreads)
end
