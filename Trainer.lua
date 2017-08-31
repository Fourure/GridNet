require 'torch'
require 'xlua'
require 'optim'

require 'MiniBatch'

do
	local Trainer = torch.class('Trainer')

	function Trainer:__init(dataset, model, criterion, batchSize, scaleMin, scaleMax, sizeX, sizeY)
		self.dataset   = dataset
		self.model     = model
		self.criterion = criterion

		self.model:cuda()
		self.criterion:cuda()

		self.batchSize = batchSize
		self.scaleMin  = scaleMin
		self.scaleMax  = scaleMax
		self.sizeX     = sizeX
		self.sizeY     = sizeY

		self.parameters, self.gradParameters = model:getParameters()
		self.confusion = optim.ConfusionMatrix(self.dataset.classes)
		self.minibatch = MiniBatch(self.dataset, self.batchSize, self.scaleMin, self.scaleMax, self.sizeX, self.sizeY)
	end

	function Trainer:setAdamParam(learningRate, learningRateDecay, epsilon, beta1, beta2)
		self.optimState = {
			learningRate 		= learningRate,
			learningRateDecay 	= learningRateDecay,
			epsilon 		= epsilon,
			beta1 			= beta1,
			beta2	 		= beta2
		}
		self.optimMethod = optim.adam
		print("Adam parameters:")
		print(self.optimState)
	end

	-- Viewed the output/target in order to feed the confusion matrix
	local transpose = function(input)
		local res = res or input:new()
		res:resizeAs(input):copy(input)
		res = res:transpose(2,4):transpose(2,3):contiguous() -- bdhw -> bwhd -> bhwd
		res = res:view(res:size(1)*res:size(2)*res:size(3), res:size(4)):contiguous()
		return res
	end

	local transpose_back = function(input, grad)
		local res = res or grad:new()
		res:resizeAs(grad):copy(grad)
		res = res:view(input:size(1),input:size(3), input:size(4), input:size(2))
		res = res:transpose(2,3):transpose(2,4):contiguous() -- bhwd -> bwhd -> bdhw
		return res
	end

	function Trainer:train(nbIteration, extra_ratio)

		local time = sys.clock()

		self.confusion:zero()
		self.model:training()
		collectgarbage()

		epoch = epoch or 1

		print("==> Doing epoch on training data:")
		print(string.format('==> epoch #%04d [batchSize =  %d]', epoch, self.batchSize))
		for t=1, nbIteration do
		
			if not opt.silent then
				xlua.progress(t, nbIteration)
			end

			-- Create mini-batch
			self.batch = self.minibatch:getTrainingBatch(extra_ratio)

			local feval = function()
				-- reset gradients
				self.gradParameters:zero()

				local outputs = self.model:forward(self.batch.inputs)

				local t_outputs = transpose(outputs)
				local t_targets = self.batch.targets:view(-1):contiguous()

				local f = self.criterion:forward(t_outputs,t_targets)
				local df_do = self.criterion:backward(t_outputs,t_targets)

				local t_df_do = transpose_back(outputs, df_do)
				self.model:backward(self.batch.inputs,t_df_do)

				self.confusion:batchAdd(t_outputs,t_targets)

				return f,self.gradParameters
			end

			-- optimize on current mini-batch
			self.optimMethod(feval, self.parameters, self.optimState)
		end

		time = sys.clock() - time
		print(string.format('\tTime : %s', xlua.formatTime(time)))

		return self.confusion
	end


	function Trainer:valid(nbIteration)

		local time = sys.clock()

		self.confusion:zero()
		self.model:evaluate()
		collectgarbage()

		epoch = epoch or 1

		print("==> Doing epoch on validation data:")
		print(string.format('==> epoch #%04d [batchSize =  %d]', epoch, self.batchSize))
		for t=1, nbIteration do

			if not opt.silent then
				xlua.progress(t, nbIteration)
			end

			-- Create mini-batch
			self.batch = self.minibatch:getValidationBatch()

			local outputs = self.model:forward(self.batch.inputs)

			self.confusion:batchAdd(transpose(outputs),self.batch.targets:view(-1))
		end

		time = sys.clock() - time
		print(string.format('\tTime : %s', xlua.formatTime(time)))

		return self.confusion
	end

end
