using DelimitedFiles, Statistics, Random, Plots, Flux, JLD2, Images, Flux.Losses
using Random:seed!
using Flux: params
using PrettyTables
using ScikitLearn
using LinearAlgebra

include("fonts/codigo_practicas.jl")
include("fonts/load_images.jl")


#Obtenemos la media y la desviacion típica de los valores RGB de las imágenes, no usado en la esta aproximacion
function meanStdDatasetRGB(dataset::Tuple{Vector{Array{Float64, 3}}, Vector{Matrix{Float64}}, BitVector})

    # Obtenemos medias y desviaciones típicas de cada imagen
    mean_red   = map(x -> mean(x[:,:,1]), dataset[1]);
    mean_green = map(x -> mean(x[:,:,2]), dataset[1]);
    mean_blue  = map(x -> mean(x[:,:,3]), dataset[1]);

    std_red   = map(x -> std(x[:,:,1]), dataset[1]);
    std_green = map(x -> std(x[:,:,2]), dataset[1]);
    std_blue  = map(x -> std(x[:,:,3]), dataset[1]);

    # Características:
    inputs = [mean_red; mean_green; mean_blue; std_red; std_green; std_blue];
    return inputs;

end

#Obtenemos la media y la desviacion típica de los valores RGB del centro de las imágenes
function meanStdCenterRGB(dataset::Tuple{Vector{Array{Float64, 3}}, Vector{Matrix{Float64}}, BitVector})

    # Obtenemos medias y desviaciones típicas de cada imagen
    mean_red   = map(x -> mean(x[8:13,8:13,1]), dataset[1]);
    mean_green = map(x -> mean(x[8:13,8:13,2]), dataset[1]);
    mean_blue  = map(x -> mean(x[8:13,8:13,3]), dataset[1]);

    std_red   = map(x -> std(x[8:13,8:13,1]), dataset[1]);
    std_green = map(x -> std(x[8:13,8:13,2]), dataset[1]);
    std_blue  = map(x -> std(x[8:13,8:13,3]), dataset[1]);

    # Características:
    inputs = [mean_red; mean_green; mean_blue; std_red; std_green; std_blue];
    return inputs;

end

function diagonal(x::Array{T,2}, k::Int) where T
    result = T[]
    for i in 1:size(x,1)
        j = i + k
        if 1 <= j <= size(x,2)
            push!(result, x[i,j])
        end
    end
    return result
end

function meanStdDatasetRGB3Lines(dataset::Tuple{Vector{Array{Float64, 3}}, Vector{Matrix{Float64}}, BitVector})
    # Coge la media y la std de cada vertical, horizontal y diagonal de la imagen en RGB
    mean_red_diagonal1 = map(x -> mean([diagonal(x[:,:,1],-1); diagonal(x[:,:,1],0); diagonal(x[:,:,1],1)]), dataset[1])
    mean_green_diagonal1 = map(x -> mean([diagonal(x[:,:,2],-1); diagonal(x[:,:,2],0); diagonal(x[:,:,2],1)]), dataset[1])
    mean_blue_diagonal1 = map(x -> mean([diagonal(x[:,:,3],-1); diagonal(x[:,:,3],0); diagonal(x[:,:,3],1)]), dataset[1])
    std_red_diagonal1 = map(x -> std([diagonal(x[:,:,1],-1); diagonal(x[:,:,1],0); diagonal(x[:,:,1],1)]), dataset[1])
    std_green_diagonal1 = map(x -> std([diagonal(x[:,:,2],-1); diagonal(x[:,:,2],0); diagonal(x[:,:,2],1)]), dataset[1])
    std_blue_diagonal1 = map(x -> std([diagonal(x[:,:,3],-1); diagonal(x[:,:,3],0); diagonal(x[:,:,3],1)]), dataset[1])

    mean_red_diagonal2 = map(x -> mean([diagonal(reverse(x[:,:,1],dims=1),-1); diagonal(reverse(x[:,:,1],dims=1), 0); diagonal(reverse(x[:,:,1],dims=1), 1)]), dataset[1])
    mean_green_diagonal2 = map(x -> mean([diagonal(reverse(x[:,:,2],dims=1),-1); diagonal(reverse(x[:,:,2],dims=1),0); diagonal(reverse(x[:,:,2],dims=1), 1)]), dataset[1])
    mean_blue_diagonal2 = map(x -> mean([diagonal(reverse(x[:,:,3],dims=1),-1); diagonal(reverse(x[:,:,3],dims=1),0); diagonal(reverse(x[:,:,3],dims=1), 1)]), dataset[1])
    std_red_diagonal2 = map(x -> std([diagonal(reverse(x[:,:,1],dims=1),-1); diagonal(reverse(x[:,:,1],dims=1),0); diagonal(reverse(x[:,:,1],dims=1), 1)]), dataset[1])
    std_green_diagonal2 = map(x -> std([diagonal(reverse(x[:,:,2],dims=1),-1); diagonal(reverse(x[:,:,2],dims=1),0); diagonal(reverse(x[:,:,2],dims=1), 1)]), dataset[1])
    std_blue_diagonal2 = map(x -> std([diagonal(reverse(x[:,:,3],dims=1),-1); diagonal(reverse(x[:,:,3],dims=1),0); diagonal(reverse(x[:,:,3],dims=1), 1)]), dataset[1])
    
    mean_red_vert   = map(x -> mean(x[10:11,:,1]), dataset[1]);
    mean_green_vert = map(x -> mean(x[10:11,:,2]), dataset[1]);
    mean_blue_vert  = map(x -> mean(x[10:11,:,3]), dataset[1]);

    std_red_vert   = map(x -> std(x[10:11,:,1]), dataset[1]);
    std_green_vert = map(x -> std(x[10:11,:,2]), dataset[1]);
    std_blue_vert  = map(x -> std(x[10:11,:,3]), dataset[1]);

    mean_red_h   = map(x -> mean(x[:,10:11,1]), dataset[1]);
    mean_green_h = map(x -> mean(x[:,10:11,2]), dataset[1]);
    mean_blue_h  = map(x -> mean(x[:,10:11,3]), dataset[1]);

    std_red_h   = map(x -> std(x[:,10:11,1]), dataset[1]);
    std_green_h = map(x -> std(x[:,10:11,2]), dataset[1]);
    std_blue_h  = map(x -> std(x[:,10:11,3]), dataset[1]);

    inputs = [mean_red_diagonal1; mean_green_diagonal1; mean_blue_diagonal1; std_red_diagonal1; std_green_diagonal1; std_blue_diagonal1;
              mean_red_diagonal2; mean_green_diagonal2; mean_blue_diagonal2; std_red_diagonal2; std_green_diagonal2; std_blue_diagonal2;
              mean_red_vert; mean_green_vert; mean_blue_vert; std_red_vert; std_green_vert; std_blue_vert;
              mean_red_h; mean_green_h; mean_blue_h; std_red_h; std_green_h; std_blue_h]
return inputs;
end;

#Fijamos la semilla aleatoria para asegurar que los experimentos son repetibles
seed!(1);

#Elegimos el número de folds para la validación cruzada
numfolds = 10;

#Establecemos los ratios de validacion y test
validationRatio = 0.2;

#Creamos los parámetros de la RNA -> en forma de vector, para dejar ejecutando los distintos modelos.
topologies_vector = [[5], [8, 4], [6, 3], [12], [4, 8], [5], [4], [6]];
maxEpochs_vector = [800, 800, 800, 800, 800, 500, 800, 800];
learningRate_vector = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01];
maxEpochsVal = 15;
numRepetitionsANNTraining = 50;
minLoss = 0.0;

#Creamos los parámetros del SVM
kernel = ["rbf", "rbf", "rbf", "linear", "poly", "poly", "rbf", "poly"];
degreeKernel = [3, 3, 3, 3, 3, 3, 3, 4]; # para los kernel distintos de 'poly' se ignorará
gammaKernel = [2, "auto", "scale", 2, 2, 2, "auto", 2];
C = [1, 1, 1, 1, 1, 10, 10, 1];

#Creamos los parametros del Decision Tree
maxDepth = [2, 4, 8, 16, 32, 64];

#Creamos los parámetros del kNN
kValue = [2, 3, 5, 10, 15, 30];

#Cargamos el dataset
dataset = loadDataset();
dataset = reshape([meanStdDatasetRGB(dataset);meanStdCenterRGB(dataset);meanStdDatasetRGB3Lines(dataset); dataset[3]], (2000, :));

#Separamos las entradas y las salidas deseadas.
inputs = Float32.(dataset[:,1:36]);
targets = dataset[:,37];        

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas";


#Generamos el vector de índices
indexVector = crossvalidation(targets, numfolds);


############ ANN ############
# Creación y entrenamiento de cada ANN
println("\n\n\n###################################");
println("###             ANN             ###")
println("###################################");

for i in eachindex(topologies_vector)
    # Imprimimos por pantalla los parámetros que estamos usando:
    print("\n\n**********************************\nANN model number $(i). Parameters:\n");

    println("Topology: $(topologies_vector[i])");
    println("MaxEpochs: $(maxEpochs_vector[i])");
    println("Learning rate: $(learningRate_vector[i])\n");

    #Pasamos los parámetros dependientes de la RNA 
    modelHyperparametersANN = Dict();
    modelHyperparametersANN["topology"] = topologies_vector[i];
    modelHyperparametersANN["validationRatio"] = validationRatio;
    modelHyperparametersANN["maxEpochs"] = maxEpochs_vector[i];
    modelHyperparametersANN["learningRate"] = learningRate_vector[i];
    modelHyperparametersANN["maxEpochsVal"] = maxEpochsVal;
    modelHyperparametersANN["minLoss"] = minLoss;
    modelHyperparametersANN["numExecutions"] = numRepetitionsANNTraining;

    #Entrenamos la RNA
    modelCrossValidation(:ANN, modelHyperparametersANN, inputs, targets, indexVector);
end


############ SVM ############
# Creación y entrenamiento de cada SVM
println("\n\n\n###################################");
println("###             SVM             ###")
println("###################################");

for i in eachindex(kernel)
    # Imprimimos por pantalla los parámetros que estamos usando:
    print("\n\n**********************************\nSVM model number $(i). Parameters:\n");

    kern = kernel[i];
    println("Kernel: $kern");
    if (kern=="poly")
        println("Degree of the kernel: $(degreeKernel[i])");
    end
    println("Gamma of the kernel: $(gammaKernel[i])");
    println("C value: $(C[i])\n");


    #Pasamos los parámetros dependientes del SVM
    modelHyperparametersSVM = Dict();
    modelHyperparametersSVM["kernel"] = kernel[i];
    modelHyperparametersSVM["kernelDegree"] = degreeKernel[i];
    modelHyperparametersSVM["kernelGamma"] = gammaKernel[i];
    modelHyperparametersSVM["C"] = C[i];

    #Entrenamos el SVM
    modelCrossValidation(:SVM, modelHyperparametersSVM, inputs, targets, indexVector);
end


############ Decision Tree ############
# Creación y entrenamiento de cada Decision Tree
println("\n\n\n###################################");
println("###        Decision Tree        ###")
println("###################################");

for i in eachindex(maxDepth)
    # Imprimimos por pantalla los parámetros que estamos usando:
    print("\n\n**********************************\nDecision Tree model number $(i). Parameters:\n");

    println("Maximum depth: $(maxDepth[i])");

    #Entrenamos el Decision Tree 
    modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth[i]), inputs, targets, indexVector);
end

############ kNN ############
# Creación y entrenamiento de cada kNN
println("\n\n\n###################################");
println("###             kNN             ###")
println("###################################");

for i in eachindex(kValue)
    # Imprimimos por pantalla los parámetros que estamos usando:
    print("\n\n**********************************\nkNN model number $(i). Parameters:\n");

    println("Number of neighbors: $(kValue[i])");

    #Entrenamos el kNN
    modelCrossValidation(:kNN, Dict("numNeighbors" => kValue[i]), inputs, targets, indexVector);
end


