require 'torch'
require 'nn'
require 'image'
require 'functions'
require 'rnn'
require 'nngraph'
require 'dpnn'

if not opt then
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Cityscapes Dataset Evaluation')
	cmd:text()
	cmd:text('Options:')
	cmd:text()
	cmd:text('Data options:')
	cmd:option('-trainLabel',false,'The model is trained with the evaluated labels only (19 classes) instead of all labels (33 classes)')
	cmd:option('-sizeX', 400, 'Input image width')
	cmd:option('-sizeY', 400, 'Input image height')
	cmd:option('-stepX', 300, 'Step for the patch')
	cmd:option('-stepY', 300, 'Step for the patch')
	cmd:option('-rgb',false,'Save the predicted images in color for visualisation')
	cmd:option('-alpha',0.5,'Transparency factor for the prediction and rgb images')
	cmd:text()
	cmd:text('Folders')
	cmd:option('-folder','','Folder in which the images are')
	cmd:option('-val',false,'Process the cityscapes\'s validation images') 
	cmd:option('-test',false,'Process the cityscapes\'s test images') 
	cmd:option('-train',false,'Process the cityscapes\'s training images') 
	cmd:text()
	cmd:text('Model options:')
	cmd:option('-model','','Trained model file')
	cmd:text()
	cmd:text('Others :')
	cmd:option('-save', '', 'subdirectory to save/log the results in')
	cmd:option('-silent',false,'Print nothing on the standards output')
	cmd:text()
	cmd:text('GPU Options :')
	cmd:option('-device',1, 'Wich GPU device to use')
	opt = cmd:parse(arg or {})

	if opt.silent then
		cmd:silent()
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
end

prediction_folder="Predictions"
image_folder= "Images"
images_paths = paths.concat(opt.save,image_folder)
predictions_paths = paths.concat(opt.save,prediction_folder)
paths.mkdir(images_paths)
paths.mkdir(predictions_paths)

if opt.trainLabel then
	classes = {	[1] = 'road',		[2]  = 'sidewalk',	[3]  = 'building',	[4]  = 'wall',
			[5] = 'fence',		[6]  = 'pole', 		[7]  = 'traffic light',	[8]  = 'traffic sign',
			[9] = 'vegetation',	[10] = 'terrain',	[11] = 'sky',		[12] = 'person',
			[13] = 'rider',		[14] = 'car',		[15] = 'truck',		[16] = 'bus',
			[17] = 'train',		[18] = 'motorcycle',	[19] = 'bicycle'
}
else
	classes = {	[1] = 'ego vehicle',		[2]  = 'rectification border',	[3]  = 'out of roi',	[4]  = 'static',
	[5] = 'dynamic',		[6]  = 'ground', 		[7]  = 'road',		[8]  = 'sidewalk',
	[9] = 'parking',		[10] = 'rail track',		[11] = 'building',	[12] = 'wall',
	[13] = 'fence',			[14] = 'guard rail',		[15] = 'bridge',	[16] = 'tunnel',
	[17] = 'pole',			[18] = 'polegroup',		[19] = 'traffic light',	[20] = 'traffic sign',
	[21] = 'vegetation',		[22] = 'terrain',		[23] = 'sky',		[24] = 'person',
	[25] = 'rider',			[26] = 'car',			[27] = 'truck',		[28] = 'bus',
	[29] = 'caravan',		[30] = 'trailer',		[31] = 'train',		[32] = 'motorcycle',
	[33] = 'bicycle'
}
end

print ("=>load model")
assert(paths.filep(opt.model),"filename " .. opt.model .. " do not refer to an existing file")
local tmp = torch.load(opt.model)
if tmp.model then
	print("Load all")
	model = tmp.model
else
	model = tmp
end

assert(model,"Error undefined model")
print("Model :")
print(model)
model:cuda()

print '==> defining confusion matrix'
confusion = optim.ConfusionMatrix(classes)

ignore_class = { 1,2,3,4,5,6,9,10,14,15,16,18,29,30 }
id_class = {[1] = 7, [2] = 8, [3] = 11, [4] = 12, [5] = 13, [6] = 17, [7] = 19, [8] = 20, [9] = 21, [10] = 22,
		[11] = 23, [12] = 24, [13] = 25, [14] = 26, [15] = 27, [16] = 28, [17] = 31, [18] = 32, [19] = 33, } 

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
   [-1] ={  0,  0,142}}


function predictionToId(prediction)

	-- If the network is trained with all the cityscapes classes we need to ignore the non-evaluated ones.
	if not opt.trainLabel then
		for k,v in pairs(ignore_class) do
			prediction[{{},v,{},{}}]:fill(-1000)
		end
	end

	local _, indMax = prediction:max(1)
	
	-- If the network is trained with the 19 classes only we need to put the evaluation indice
	if opt.trainLabel then
		indMax:apply(function(x) return id_class[x] end)
	end

	return indMax
end

confusion:zero()
model:evaluate()

function test(folder)

	print("Looking for files in " .. folder)

	for f in paths.iterdirs(folder) do
		test(paths.concat(folder, f))
	end

	print("==> Processing data for test:")
	for file in paths.iterfiles(folder) do

		print("Processing: " .. file)

		img = image.load(folder .. "/" .. file)

		if not img then
			print("<Warning: cannot load image " .. filename .. ">")
		else
			
			batch = batch or {}
			batch.inputs   = batch.inputs    or torch.Tensor(1, 3, opt.sizeY, opt.sizeX)

			prediction = torch.Tensor(#classes, img:size(2), img:size(3)):fill(0)
			prediction = prediction:float()

			batch.inputs  = batch.inputs:cuda()

			test_scale = {-2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5}
			for k,s in ipairs(test_scale) do
				for y=0, img:size(2)-1, opt.stepY do
					for x=0, img:size(3)-1, opt.stepX do
						--Crop image
						scale = s
						hflip=false

						if s < 0 then
							scale = -s
							hflip = true
						end

						sX = x
						sY = y
						eX = x + opt.sizeX*scale
						eY = y + opt.sizeY*scale
						if eX > img:size(3) then
							eX = img:size(3)
							sX = eX - opt.sizeX*scale
						end

						if eY > img:size(2) then
							eY = img:size(2)
							sY = eY - opt.sizeY*scale
						end

						inputCrop = image.crop(img,sX,sY,eX,eY)
						if hflip then
							inputCrop = image.hflip(inputCrop)
						end

						batch.inputs[1]:copy(image.scale(inputCrop, opt.sizeX, opt.sizeY))

						local output = model:forward(batch.inputs)

						output = image.scale(output[1]:float(), eX-sX, eY-sY)

						if hflip then
							output = image.hflip(output)
						end

						prediction[{{},{sY+1,eY},{sX+1,eX}}]:add(output)

						local transpose = function(input)
							input = input:transpose(2,4):transpose(2,3):contiguous() -- bdhw -> bwhd -> bhwd
							input = input:view(input:size(1)*input:size(2)*input:size(3), input:size(4))
							return input
						end
					end
				end
			end

			idPrediction = predictionToId(prediction)

			if opt.rgb then
				rgb = torch.repeatTensor(idPrediction,3,1,1)

				for i=1, 3 do
					rgb[i]:apply(function(x)
						return IdToRGB[x][i]
					end)
				end

				rgb = rgb:float()
				img = img:float()
				img:mul(255)

				rgb:mul(opt.alpha):add(img:mul(1-opt.alpha))

				filename = paths.concat(opt.save, image_folder, paths.basename(file))
				image.save(filename, rgb:float():div(255))
			end


			filename = paths.concat(opt.save, prediction_folder, paths.basename(file))
			image.save(filename, idPrediction:float():div(255))
		end
	end
end

dataset_folder	 = os.getenv("CITYSCAPES_DATASET") or paths.cwd()
raw_folder 	 = "leftImg8bit"

-- Train step
if opt.val then
	local directory = paths.concat(dataset_folder,raw_folder, "val")
	time(test,directory)
end
if opt.test then
	local directory = paths.concat(dataset_folder,raw_folder, "test")
	time(test,directory)
end
if opt.train then
	local directory = paths.concat(dataset_folder,raw_folder, "train")
	time(test,directory)
end
if opt.folder ~= '' then
	time(test,opt.folder)
end
