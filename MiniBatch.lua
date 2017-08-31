require 'image'

do
	local MiniBatch = torch.class('MiniBatch')

	function MiniBatch:__init(dataset, batchSize, scaleMin, scaleMax, sizeX, sizeY, hflip)
		self.dataset   = dataset
		self.batchSize = batchSize
		self.scaleMin  = scaleMin
		self.scaleMax  = scaleMax
		self.sizeX     = sizeX
		self.sizeY     = sizeY
		self.hflip     = hflip

		self.batch = {}
		self.batch.inputs  = torch.Tensor(self.batchSize, 3, self.sizeY, self.sizeX)
		self.batch.targets = torch.Tensor(self.batchSize, self.sizeY, self.sizeX)

		self.batch.inputs  = self.batch.inputs:cuda()
		self.batch.targets = self.batch.targets:cuda()
	end

	function MiniBatch:preprocess(input, target)
		local scale_factor = torch.uniform(self.scaleMin, self.scaleMax)

		target = torch.squeeze(target)

		local input_sizeX = input:size(3)
		local input_sizeY = input:size(2)

		local crop_x = self.sizeX * scale_factor
		local crop_y = self.sizeY * scale_factor

		--print("Size crop X : " .. crop_x)
		--print("Size crop Y : " .. crop_y)

		local offsetX = 0
		local offsetY = 0

		if input_sizeX ~= crop_x then
			offsetX = (torch.random()%(input_sizeX-crop_x))
		end
		if input_sizeY ~= crop_y then
			offsetY = (torch.random()%(input_sizeY-crop_y))
		end

		local input_cropped  = image.crop(input , offsetX, offsetY, offsetX + crop_x, offsetY + crop_y)
		local target_cropped = image.crop(target, offsetX, offsetY, offsetX + crop_x, offsetY + crop_y)

		local input_scaled  = image.scale(input_cropped,self.sizeX, self.sizeY)
		local target_scaled = image.scale(target_cropped, self.sizeX, self.sizeY, "simple")

		if self.hflip and torch.random()%2 == 0 then
			input_scaled  = image.hflip(input_scaled)
			target_scaled = image.hflip(target_scaled)
		end

		return input_scaled, target_scaled
	end

	function MiniBatch:getTrainingBatch(extra_ratio)
		local ratio = extra_ratio or 0

		for i=1, self.batchSize do
			local input, target
			if ratio > 0 and torch.uniform() < ratio then
				input, target = self.dataset:next_extra_sample()
			else
				input, target = self.dataset:next_training_sample()
			end

			local preprocessed_input, preprocessed_target = self:preprocess(input,target)

			self.batch.inputs[i]:copy(preprocessed_input)
			self.batch.targets[i]:copy(preprocessed_target)
		end

		return self.batch
	end

	function MiniBatch:getValidationBatch()

		for i=1, self.batchSize do

			local input, target = self.dataset:next_validation_sample()
			local preprocessed_input, preprocessed_target = self:preprocess(input,target)

			self.batch.inputs[i]:copy(preprocessed_input)
			self.batch.targets[i]:copy(preprocessed_target)
		end

		return self.batch
	end

	function MiniBatch:saveBatch()
		for i=1, self.batchSize do
			local img = self.batch.inputs[i]
			local gt  = self.batch.targets[i]

			image.save("miniBatchInput" .. i .. ".png", img)
			image.save("miniBatchTarget" .. i .. ".png", self.dataset:gtToImage(img, gt))
		end
	end

end
