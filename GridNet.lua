require 'torch'
require 'nn'
require 'image'

require 'cunn'
require 'cudnn'

require 'dpnn'
require 'nngraph'

color = {   convolution        = "darkgoldenrod1",
            subSampling        = "darkgoldenrod", 
            fullConvolution    = "firebrick1",
            upSampling         = "firebrick",
            batchNormalization = "deepskyblue3",
            relu               = "darkolivegreen3",
            add                = "bisque3",
            dropout            = "darkviolet"}

function firstConv(input, nInputs, nOutputs)
    local seq = input

    seq = seq - cudnn.SpatialConvolution(nInputs, nOutputs, 3, 3, 1, 1, 1, 1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["convolution"]}})

    seq = seq - cudnn.SpatialBatchNormalization(nOutputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})

    seq = seq - cudnn.ReLU(true)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["relu"]}})

    seq = seq - cudnn.SpatialConvolution(nOutputs, nOutputs, 3, 3, 1, 1, 1, 1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["convolution"]}})

    seq = seq - cudnn.SpatialBatchNormalization(nOutputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})

    seq = seq - cudnn.ReLU(true)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["relu"]}})

    return seq
end

function convSequence(input, nInputs, nOutputs, dropFactor)
    local seq = input
    
    seq = seq - cudnn.SpatialBatchNormalization(nInputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})
    
    seq = seq  - cudnn.ReLU(true)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["relu"]}})
    
    seq = seq - cudnn.SpatialConvolution(nInputs, nOutputs, 3, 3, 1, 1, 1, 1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["convolution"]}})
    
    seq = seq - cudnn.SpatialBatchNormalization(nOutputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})
    
    seq = seq  - cudnn.ReLU(true)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["relu"]}})
    
    seq = seq - cudnn.SpatialConvolution(nOutputs, nOutputs, 3, 3, 1, 1, 1, 1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["convolution"]}})
    
    seq = seq - nn.TotalDropout(dropFactor)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["dropout"]}})
    
    return seq
end

function lastDeconv(input, nInputs, nOutputs)
    local seq = input 
    
    seq = seq - cudnn.SpatialFullConvolution(nInputs, nOutputs, 3, 3, 1, 1, 1, 1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["fullConvolution"]}})

    seq = seq - cudnn.SpatialBatchNormalization(nOutputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})

    seq = seq - cudnn.ReLU(true)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["relu"]}})

    seq = seq - cudnn.SpatialFullConvolution(nOutputs, nOutputs, 3, 3, 1, 1, 1, 1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["fullConvolution"]}})

    seq = seq - cudnn.SpatialBatchNormalization(nOutputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})

    return seq
end

function deconvSequence(input, nInputs, nOutputs, dropFactor)
    local seq = input 
    
    seq = seq - cudnn.SpatialBatchNormalization(nInputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})

    seq = seq - cudnn.ReLU(true)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["relu"]}})

    seq = seq - cudnn.SpatialFullConvolution(nInputs, nOutputs,3,3,1,1,1,1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["fullConvolution"]}})
    
    seq = seq - cudnn.SpatialBatchNormalization(nOutputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})

    seq = seq - cudnn.ReLU(true)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["relu"]}})

    seq = seq - cudnn.SpatialFullConvolution(nOutputs, nOutputs,3,3,1,1,1,1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["fullConvolution"]}})

    seq = seq - nn.TotalDropout(dropFactor)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["dropout"]}})
    
    return seq
end

function addTransform(convInput, poolInput, nInputs)
    local res = {poolInput, convInput} - nn.CAddTable() 
    res:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["add"]}})

    return res
end

function add3Transform(convInput, poolInput, residualInput, nInputs)
    local res = {residualInput, poolInput, convInput} - nn.CAddTable()
    res:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["add"]}})

    return res
end

function subSamplingSequence(input, nInputs, nOutputs)
    local seq = input 
    
    seq = seq - cudnn.SpatialBatchNormalization(nInputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})
    
    seq = seq - cudnn.ReLU(true)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["relu"]}})

    seq = seq - cudnn.SpatialConvolution(nInputs, nOutputs, 3, 3, 2, 2, 1, 1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["subSampling"]}})
    
    seq = seq - cudnn.SpatialBatchNormalization(nOutputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})
    
    seq = seq - cudnn.ReLU(true)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["relu"]}})

    seq = seq - cudnn.SpatialConvolution(nOutputs, nOutputs, 3, 3, 1, 1, 1, 1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["convolution"]}})

    return seq
end

function upSamplingSequence(input, nInputs, nOutputs)
    local seq = input 
    
    seq = seq - cudnn.SpatialBatchNormalization(nInputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})

    seq = seq - cudnn.ReLU(true)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["relu"]}})

    seq = seq - cudnn.SpatialFullConvolution(nInputs, nOutputs, 3, 3, 2, 2, 1, 1, 1, 1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["upSampling"]}})
    
    seq = seq - cudnn.SpatialBatchNormalization(nOutputs)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["batchNormalization"]}})

    seq = seq - cudnn.ReLU(true)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["relu"]}})

    seq = seq - cudnn.SpatialFullConvolution(nOutputs, nOutputs,3,3,1,1,1,1)
    seq:annotate({graphAttributes = {color = 'black', style = 'filled', fillcolor = color["fullConvolution"]}})

    return seq
end

function createGridNet(nInputs, nOutputs, nColumns, nFeatMaps, dropFactor)

    nStreams = #nFeatMaps

    local input = cudnn.SpatialBatchNormalization(nInputs)()

    local C = {}
    C[1] = {}

    --Create input (first feature of each streams)
    C[1][1] = firstConv(input, nInputs, nFeatMaps[1])
    for s=2, nStreams do
        C[1][s] = subSamplingSequence(C[1][s-1], nFeatMaps[s-1], nFeatMaps[s])
    end

    --Construct the conv part of each streams
    for r=2, nColumns do
        C[r] = {}
        C[r][1] = addTransform(
            convSequence(C[r-1][1], nFeatMaps[1], nFeatMaps[1], dropFactor),
            C[r-1][1],
            nFeatMaps[1]
        )
        
        for s=2, nStreams do
            C[r][s] = add3Transform(
                convSequence(C[r-1][s], nFeatMaps[s], nFeatMaps[s],dropFactor),
                subSamplingSequence(C[r][s-1],nFeatMaps[s-1],nFeatMaps[s]),
                C[r-1][s],
                nFeatMaps[s]
            )
        end
    end

    --First column of deconv
    r=nColumns
    C[r][nStreams] = addTransform(
        convSequence(C[r][nStreams], nFeatMaps[nStreams], nFeatMaps[nStreams], dropFactor),
        C[r][nStreams],
        nFeatMaps[nStreams]
    )

    ---[[
    for s=nStreams-1, 1, -1 do
        C[r][s] = add3Transform(
            convSequence(C[r][s], nFeatMaps[s], nFeatMaps[s], dropFactor),
            upSamplingSequence(C[r][s+1], nFeatMaps[s+1], nFeatMaps[s]),
            C[r][s],
            nFeatMaps[s]
        ) 
    end
    --]]

    --Construct the deconv part of each streams
    ---[[
    for r=nColumns-1, 1, -1 do
        C[r][nStreams] = addTransform(
            deconvSequence(C[r+1][nStreams], nFeatMaps[nStreams], nFeatMaps[nStreams], dropFactor),
            C[r+1][nStreams], 
            nFeatMaps[nStreams]
        )

        for s=nStreams-1, 1, -1 do
            C[r][s] = add3Transform(
                deconvSequence(C[r+1][s], nFeatMaps[s], nFeatMaps[s], dropFactor),
                upSamplingSequence(C[r][s+1], nFeatMaps[s+1], nFeatMaps[s]),
                C[r+1][s],
                nFeatMaps[s]
            )
        end
    end
    --]]

    local output = lastDeconv(C[1][1], nFeatMaps[1], nOutputs)

    local model = nn.gModule({input},{output})

    local model_parameters = {
        nfeats  = nInputs,
        noutputs = nOutputs,
        ncolumns = nColumns,
        nfeatsmaps = #nFeatMaps,
        dropfactor = dropFactor
    }

    return model, model_parameters
end

--model, model_parameters = createGridNet(3,19,3,{16,32,64,128,256},0.1)
--graph.dot(model.fg, 'Grid Network', "gridNetwork")


function test_model(batch, sizeX, sizeY)

    sizeY = sizeY or sizeX

    --criterion = cudnn.SpatialCrossEntropyCriterion()

    model, model_parameters = createGridNet(3,19,3,{8,16,32,64,128,256},0.1)
    --model:add(nn.LogSoftMax())
    criterion = nn.CrossEntropyCriterion()


    model:cuda()
    model:training()
    criterion:cuda()
    input = torch.rand(batch or 6,3,sizeX or 224,sizeY or 224):cuda()
    
    output = model:forward(input)

    print(output:size())


    target = torch.Tensor(output:size(1),output:size(3),output:size(4)):fill(5)
    target = target:random()%output:size(2)
    target:add(1)
    target = target:cuda()


    err = criterion:forward(output,target)
    df_do = criterion:backward(output,target)
    model:backward(input,df_do)

    model:updateParameters(0.01)
end
