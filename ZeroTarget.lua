require 'cudnn'

local ZeroTarget, parent = torch.class('cudnn.ZeroTarget', 'nn.Criterion')

function ZeroTarget:__init(criterion)
	parent.__init(self)
	self.criterion = criterion
	self.target = torch.Tensor()
end

local function convertMask(mask, input)
	mask = mask:view(1,mask:size(1))
	mask = mask:expand(input:size(2),input:size(1)):transpose(1,2)
	return mask
end

function ZeroTarget:updateOutput(input, target)
	assert(input:dim() == 2, 'mini-batch supported only')
	assert(target:dim() == 1, 'mini-batch supported only')
	assert(input:size(1) == target:size(1), 'input and target should be of same size')

	self.mask = torch.lt(target,1)
	self.target:resizeAs(target):copy(target):clamp(1,input:size(2))
	
	self.criterion:updateOutput(input,self.target)

	self.output = self.criterion.output
	return self.output
end

function ZeroTarget:updateGradInput(input, target)
	
	--self.mask = torch.lt(target,1)
	--self.target:resizeAs(target):copy(target):clamp(1,input:size(2))

	self.criterion:updateGradInput(input,self.target)

	self.gradInput = self.criterion.gradInput
	self.gradInput[convertMask(self.mask,input)]=0

	return self.gradInput
end

function ZeroTarget:type(type)
	if type then
		self.criterion:type(type)
		self.target:type(type)
	end

	parent.type(self, type)
	return self
end
