using DelimitedFiles, Statistics, Random, Plots, Flux, JLD2, Images, Flux.Losses
using Random:seed!
using Flux: params
using PrettyTables
using ScikitLearn
using CSV

include("fonts/codigo_practicas.jl")
include("fonts/load_images.jl")


#Obtenemos la media y la desviacion típica de los valores RGB de las imágenes
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

#Obtenemos la media y la desviacion típica de los valores RGB de las esquinas de las imágenes
function meanStdCornersRGB(dataset::Tuple{Vector{Array{Float64, 3}}, Vector{Matrix{Float64}}, BitVector})

    # Obtenemos medias y desviaciones típicas de cada imagen
    mean_red_c11   = map(x -> mean(x[1:4,1:4,1]), dataset[1]);
    mean_green_c11 = map(x -> mean(x[1:4,1:4,2]), dataset[1]);
    mean_blue_c11  = map(x -> mean(x[1:4,1:4,3]), dataset[1]);
    std_red_c11   = map(x -> std(x[1:4,1:4,1]), dataset[1]);
    std_green_c11 = map(x -> std(x[1:4,1:4,2]), dataset[1]);
    std_blue_c11  = map(x -> std(x[1:4,1:4,3]), dataset[1]);

    mean_red_c12   = map(x -> mean(x[1:4,17:20,1]), dataset[1]);
    mean_green_c12 = map(x -> mean(x[1:4,17:20,2]), dataset[1]);
    mean_blue_c12  = map(x -> mean(x[1:4,17:20,3]), dataset[1]);
    std_red_c12   = map(x -> std(x[1:4,17:20,1]), dataset[1]);
    std_green_c12 = map(x -> std(x[1:4,17:20,2]), dataset[1]);
    std_blue_c12  = map(x -> std(x[1:4,17:20,3]), dataset[1]);

    mean_red_c21   = map(x -> mean(x[17:20,1:4,1]), dataset[1]);
    mean_green_c21 = map(x -> mean(x[17:20,1:4,2]), dataset[1]);
    mean_blue_c21  = map(x -> mean(x[17:20,1:4,3]), dataset[1]);
    std_red_c21   = map(x -> std(x[17:20,1:4,1]), dataset[1]);
    std_green_c21 = map(x -> std(x[17:20,1:4,2]), dataset[1]);
    std_blue_c21  = map(x -> std(x[17:20,1:4,3]), dataset[1]);

    mean_red_c22   = map(x -> mean(x[17:20,17:20,1]), dataset[1]);
    mean_green_c22 = map(x -> mean(x[17:20,17:20,2]), dataset[1]);
    mean_blue_c22  = map(x -> mean(x[17:20,17:20,3]), dataset[1]);
    std_red_c22   = map(x -> std(x[17:20,17:20,1]), dataset[1]);
    std_green_c22 = map(x -> std(x[17:20,17:20,2]), dataset[1]);
    std_blue_c22  = map(x -> std(x[17:20,17:20,3]), dataset[1]);

    # Características:
    inputs = [mean_red_c11; mean_green_c11; mean_blue_c11; std_red_c11; std_green_c11; std_blue_c11;
              mean_red_c12; mean_green_c12; mean_blue_c12; std_red_c12; std_green_c12; std_blue_c12;
              mean_red_c21; mean_green_c21; mean_blue_c21; std_red_c21; std_green_c21; std_blue_c21;
              mean_red_c22; mean_green_c22; mean_blue_c22; std_red_c22; std_green_c22; std_blue_c22;];
    return inputs;

end

#************************************************************************************************************

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
dataset = reshape([meanStdDatasetRGB(dataset); meanStdCenterRGB(dataset); meanStdCornersRGB(dataset); dataset[3]], (2000, :));

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

ruta = "resultados/aprox4_ANN.csv";
cabecera = [("model", "topology", "maxEpochs", "learningRate", "meanPrecission", "stdPrecission", "meanF1", "stdF1")];
writedlm(ruta, cabecera, "&");
f = open(ruta, "a");

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

    #Creamos el CSV para los resultados
    parametros = ["$i", topologies_vector[i], maxEpochs_vector[i], learningRate_vector[i]];
    
    # Entrenamos la ANN y guardamos los resultados en el csv
    append!(parametros, round.(modelCrossValidation(:ANN, modelHyperparametersANN, inputs, targets, indexVector), digits=4)...);
    writedlm(f, reshape(parametros, (1,:)), "&");

end
close(f);


############ SVM ############
# Creación y entrenamiento de cada SVM
println("\n\n\n###################################");
println("###             SVM             ###")
println("###################################");

ruta = "resultados/aprox4_SVM.csv";
cabecera = [("model", "kernel", "degreeKernel", "gammaKernel", "C", "meanPrecission", "stdPrecission", "meanF1", "stdF1")];
writedlm(ruta, cabecera, "&"); # Inicia un archivo nuevo, si lo había
f = open(ruta, "a");

for i in eachindex(kernel)
    # Imprimimos por pantalla los parámetros que estamos usando:
    print("\n\n**********************************\nSVM model number $(i). Parameters:\n");

    kern = kernel[i];
    println("Kernel: $kern");
    if (kern=="poly")
        println("Degree of the kernel: $(degreeKernel[i])");
        parametros = [i, kern, degreeKernel[i], gammaKernel[i], C[i]];
    else
        parametros = [i, kern, "", gammaKernel[i], C[i]];
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
    append!(parametros, round.(modelCrossValidation(:SVM, modelHyperparametersSVM, inputs, targets, indexVector), digits=4)...);
    writedlm(f, reshape(parametros, (1,:)), "&");
end
close(f);


############ Decision Tree ############
# Creación y entrenamiento de cada Decision Tree
println("\n\n\n###################################");
println("###        Decision Tree        ###")
println("###################################");

ruta = "resultados/aprox4_DecisionTree.csv";
cabecera = [("model", "maximumDepth", "meanPrecission", "stdPrecission", "meanF1", "stdF1")];
writedlm(ruta, cabecera, "&");
f = open(ruta, "a");


for i in eachindex(maxDepth)
    # Imprimimos por pantalla los parámetros que estamos usando:
    print("\n\n**********************************\nDecision Tree model number $(i). Parameters:\n");
    println("Maximum depth: $(maxDepth[i])");

    parametros = ["$i", maxDepth[i]];

    #Entrenamos el Decision Tree 
    append!(parametros, round.(modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth[i]), inputs, targets, indexVector), digits=4)...);
    writedlm(f, reshape(parametros, (1,:)), "&");
end
close(f);


############ kNN ############
# Creación y entrenamiento de cada kNN
println("\n\n\n###################################");
println("###             kNN             ###")
println("###################################");

ruta = "resultados/aprox4_kNN.csv";
cabecera = [("model", "neighbors", "meanPrecission", "stdPrecission", "meanF1", "stdF1")];
writedlm(ruta, cabecera, "&");
f = open(ruta, "a");

for i in eachindex(kValue)
    # Imprimimos por pantalla los parámetros que estamos usando:
    print("\n\n**********************************\nkNN model number $(i). Parameters:\n");
    println("Number of neighbors: $(kValue[i])");

    parametros = ["$i", kValue[i]];

    #Entrenamos el kNN
    append!(parametros, round.(modelCrossValidation(:kNN, Dict("numNeighbors" => kValue[i]), inputs, targets, indexVector), digits=4)...);
    writedlm(f, reshape(parametros, (1,:)), "&");
    
end
close(f);
