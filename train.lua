require 'torch'
require 'xlua'
require 'optim'
require 'image'

require 'CityscapesLoader'
require 'GridNet'
require 'functions'
require 'parameters'
require 'Trainer'
require 'ZeroTarget'


dataset = CityscapesLoader()

if paths.filep(opt.model) then
	print("Load model from file : " .. opt.model)

	local tmp = torch.load(opt.model)
	model = tmp.model
	epoch = tmp.epoch
	criterion = tmp.criterion
	model_parameters= tmp.model_parameters
else
	model, model_parameters = createGridNet(3,#dataset.classes,3,{16,32,64,126,256},opt.dropFactor)
end

criterion = cudnn.ZeroTarget(nn.CrossEntropyCriterion())


print("Model used:")
print(model)
print("Criterion used:")
print(criterion)

function clearState()
	model:clearState()
end
clearState()

function saveModel(filename)
	torch.save(filename, {epoch=epoch, model=model, model_parameters=model_parameters, criterion=criterion})
end

trainer = Trainer(dataset, model, criterion, opt.batchSize, opt.scaleMin, opt.scaleMax, opt.sizeX, opt.sizeY, opt.hflip)
trainer:setAdamParam(opt.learningRate, opt.learningRateDecay, opt.epsilon, opt.beta1, opt.beta2)

local best_res = 0

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
validLogger = optim.Logger(paths.concat(opt.save, 'valid.log'))

while true do

	-- Train step
	local confusion = trainer:train(opt.nbIterationTrain, opt.extra_ratio)
	local avg_row, avg_voc, glb_cor = get_accuracy(confusion)
	trainLogger:add{['#epoch pixel class IoU'] = epoch .. " " .. glb_cor .. " " .. avg_row .. " " .. avg_voc }

	-- Validation step
	local confusion = trainer:valid(opt.nbIterationValid)
	local avg_row, avg_voc, glb_cor = get_accuracy(confusion)
	validLogger:add{['#epoch pixel class IoU'] = epoch .. " " .. glb_cor .. " " .. avg_row .. " " .. avg_voc }

	clearState()

	-- Save model if better
	if avg_voc > best_res then
		local filename_best = paths.concat(opt.save, 'best_model.t7')
		saveModel(filename_best)
		best_res = avg_voc
	end

	-- Save the last model at each epoch
	local filename_last = paths.concat(opt.save, 'last_model.t7')
	saveModel(filename_last)

	epoch = epoch + 1
end
