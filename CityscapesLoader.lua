require 'image'

do
	local CityscapesSet = torch.class('CityscapesSet')

	function CityscapesSet:__init()
		self.image = {}
		self.gt    = {}
		self.current_indice = 0
	end

	function CityscapesSet:size()
		return #self.image
	end

	function CityscapesSet:addSample(imagePath, groundTruthPath)
		table.insert(self.image,imagePath)
		table.insert(self.gt, groundTruthPath)
	end

	function CityscapesSet:load_next(dataset_folder)
		local img, gt
		
		if self.current_indice == self:size() then
			self.current_indice = 0
			self:initShuffle()
		end

		self.current_indice = self.current_indice + 1

		img = image.load(paths.concat(dataset_folder, self.image[self.shuffle[self.current_indice]]))
		gt  = image.load(paths.concat(dataset_folder, self.gt[self.shuffle[self.current_indice]]))
		gt:mul(255)

		return img, gt
	end

	function CityscapesSet:initShuffle()
		self.shuffle = torch.randperm(self:size())
	end
end



do
	local CityscapesLoader = torch.class('CityscapesLoader')

	function CityscapesLoader:__init()
		self.dataset_folder 	= os.getenv("CITYSCAPES_DATASET") or paths.cwd()
		self.raw_folder 	= "leftImg8bit"
		self.gt_fine_folder 	= "gtFine"
		self.gt_coarse_folder 	= "gtCoarse"
		self.gt_type 		= "_labelTrainIds"

		self.train_set       = CityscapesSet()
		self.extra_train_set = CityscapesSet()
		self.val_set         = CityscapesSet()
		self.test_set        = CityscapesSet()

		self.classes = {	[1] = 'road',		[2]  = 'sidewalk',	[3]  = 'building',	[4]  = 'wall',
			[5] = 'fence',		[6]  = 'pole', 		[7]  = 'traffic light',	[8]  = 'traffic sign',
			[9] = 'vegetation',	[10] = 'terrain',	[11] = 'sky',		[12] = 'person',
			[13] = 'rider',		[14] = 'car',		[15] = 'truck',		[16] = 'bus',
			[17] = 'train',		[18] = 'motorcycle',	[19] = 'bicycle'
		}


		print "--> Loading train set"
		self:load_split(self.train_set, "train")
		print "--> Loading extra train data"
		self:load_split(self.extra_train_set, "train_extra", true)
		print "--> Loading test set"
		self:load_split(self.test_set , "test" )
		print "--> Loading validation set"
		self:load_split(self.val_set  , "val"  )

		self.train_set:initShuffle()
		self.extra_train_set:initShuffle()
		self.val_set:initShuffle()
		self.test_set:initShuffle()

		print("\tTrain size      : " .. self.train_set:size())
		print("\tExtra train size: " .. self.extra_train_set:size())
		print("\tTest  size      : " .. self.test_set:size())
		print("\tValidation size : " .. self.val_set:size())
	end

	function CityscapesLoader:load_split( res_set, split, extra_data )
		local directory = paths.concat(self.dataset_folder, self.raw_folder, split)
		assert(paths.dirp(directory),"Cannot find split " .. split .. " into '" .. self.raw_folder .. "'")

		local gt_folder = self.gt_fine_folder
		if extra_data then
			gt_folder = self.gt_coarse_folder
		end

		for city in paths.iterdirs(directory) do
			local city_folder = paths.concat(directory, city)

			for file in paths.iterfiles(city_folder) do
				local gt_file = file:gsub(self.raw_folder, gt_folder .. self.gt_type)

				file    = self.raw_folder .. "/" .. split .. "/" .. city .. "/" .. file
				gt_file = gt_folder  .. "/" .. split .. "/" .. city .. "/" .. gt_file

				res_set:addSample(file, gt_file)
			end
		end
	end

	function CityscapesLoader:next_training_sample()
		return self.train_set:load_next(self.dataset_folder)
	end

	function CityscapesLoader:next_extra_sample()
		return self.extra_train_set:load_next(self.dataset_folder)
	end

	function CityscapesLoader:next_validation_sample()
		return self.val_set:load_next(self.dataset_folder)
	end

	function CityscapesLoader:next_test_sample()
		return self.test_set:load_next(self.dataset_folder)
	end

	function CityscapesLoader:gtToImage(input, gt)

		IdToRGB = {
			[0] ={  0,  0,  0},
			[1] ={  0,  0,  0},
			[2] ={  0,  0,  0},
			[3] ={  0,  0,  0},
			[4] ={  0,  0,  0},
			[5] ={111, 74,  0},
			[6] ={ 81,  0, 81},
			[7] ={128, 64,128},
			[8] ={244, 35,232},
			[9] ={250,170,160},
			[10] ={230,150,140},
			[11] ={ 70, 70, 70},
			[12] ={102,102,156},
			[13] ={190,153,153},
			[14] ={180,165,180},
			[15] ={150,100,100},
			[16] ={150,120, 90},
			[17] ={153,153,153},
			[18] ={153,153,153},
			[19] ={250,170, 30},
			[20] ={220,220,  0},
			[21] ={107,142, 35},
			[22] ={152,251,152},
			[23] ={ 70,130,180},
			[24] ={220, 20, 60},
			[25] ={255,  0,  0},
			[26] ={  0,  0,142},
			[27] ={  0,  0, 70},
			[28] ={  0, 60,100},
			[29] ={  0,  0, 90},
			[30] ={  0,  0,110},
			[31] ={  0, 80,100},
			[32] ={  0,  0,230},
			[33] ={119, 11, 32},
			[-1] ={  0,  0,142}
		}

		rgb = torch.repeatTensor(gt,3,1,1)

		for i=1, 3 do
			rgb[i]:apply(function(x)
				--return IdToRGB[torch.floor(x+0.5)][i]
				return IdToRGB[x][i]
			end)
		end

		ratio = 0.60
		rgb = rgb:float():mul(ratio)
		original = input:mul(255):float():mul(1-ratio)
		rgb:add(original)

		return rgb:div(255)
	end
	--End do
end
