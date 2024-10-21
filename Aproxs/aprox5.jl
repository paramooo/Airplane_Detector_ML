
using Flux: onehotbatch, onecold, params
using JLD2, FileIO
using DelimitedFiles, Statistics, Random, Plots, Flux, Flux.Losses
using Statistics:mean
using Random:seed!
using PrettyTables
using ScikitLearn

include("fonts/load_images.jl")
include("fonts/codigo_practicas.jl")


# Comenzamos definiendo las funciones que crearán las CNN que usaremos

function createCNN_1()
    return Chain(
        Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(128, 1, σ),
    );
end;

function createCNN_2()
    return Chain(
        Conv((5, 5), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((5, 5), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        Conv((5, 5), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        x -> reshape(x, :, size(x, 4)),
        Dense(800, 1, σ),
    );
end;

function createCNN_3()
    return Chain(
        Conv((2, 2), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((2, 2), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((2, 2), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(288, 1, σ),
    );
end;

function createCNN_4()
    return Chain(
        Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        x -> reshape(x, :, size(x, 4)),
        Dense(800, 1, σ),
    );
end;

function createCNN_5()
    return Chain(
        Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(800, 1, σ),
    );
end;

function createCNN_6()
    return Chain(
        Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 32=>64, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(256, 1, σ),
    );
end;

function createCNN_7()
    return Chain(
        Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 32=>64, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        x -> reshape(x, :, size(x, 4)),
        Dense(1600, 1, σ),
    );
end;

function createCNN_8()
    return Chain(
        Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(1600, 1, σ),
    );
end;

function createCNN_3x3()
    return Chain(
        Conv((3, 3), 3 => 16, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 16 => 32, relu),
        MaxPool((2, 2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(288, 1, σ)
    )
end


function createCNN_small()
    return Chain(
        Conv((3, 3), 3=>8, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        MaxPool((2,2)),
        Conv((3, 3), 8=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        x -> reshape(x, :, size(x, 4)),
        Dense(1600, 1, σ),
    );
end;

function createCNN_simple1()
    return Chain(
        Conv((3, 3), 3=>8, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        x -> reshape(x, :, size(x, 4)),
        Dense(800, 1, σ),
    );
end;

function createCNN_simple2()
    return Chain(
        Conv((5, 5), 3=>8, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        x -> reshape(x, :, size(x, 4)),
        Dense(800, 1, σ),
    );
end;

function createCNN_simple3()
    return Chain(
        Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
        x -> reshape(x, :, size(x, 4)),
        Dense(1600, 1, σ),
    );
end;



### Código Deep Learning

function convertirArrayImagenesHWCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, 20, 20, 3, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(20,20,3)) "Las imagenes no tienen tamaño 20x20";
        for j in 1:3
            nuevoArray[:,:,j,i] .= imagenes[i][:,:,j];
        end;
    end;
    return nuevoArray;
end;


# Creación y entrenamiento CNN
function trainClassCNN(train_set::Tuple{AbstractArray{<:Real,4}, AbstractArray{Bool,2}}, 
                        test_set::Tuple{AbstractArray{<:Real,4}, AbstractArray{Bool,2}},
                        generator::Function) # añadir topology o similar

    (inputs, targets) = train_set;

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide
    @assert(size(inputs,4)==size(targets,1));

    # Creamos la CNN
    ann = generator();

    # Definimos la funcion de loss
    loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    # println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*(accuracy(ann(inputs)', targets)), " %");

    # println("Comenzando entrenamiento...")

    opt = ADAM(0.001);
    mejorPrecision = -Inf;
    criterioFin = false;
    numCiclo = 0;
    numCicloUltimaMejora = 0;
    # mejorModelo = nothing;

    while (!criterioFin)
        
        # Entrenamos
        Flux.train!(loss, params(ann), [(inputs, targets')], opt);

        numCiclo += 1;

        precisionEntrenamiento = accuracy(ann(inputs)', targets);
        # println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

        # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
        if (precisionEntrenamiento >= mejorPrecision)
            mejorPrecision = precisionEntrenamiento;
            # precisionTest = accuracy(ann(test_set[1])', test_set[2]);
            # println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
            # mejorModelo = deepcopy(ann);
            numCicloUltimaMejora = numCiclo;
        end

        # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
        if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
            opt.eta /= 10.0
            # println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta);
            numCicloUltimaMejora = numCiclo;
        end


            # Criterios de parada:
        # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
        if (precisionEntrenamiento >= 0.999)
            # println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
            criterioFin = true;
        end

        # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
        if (numCiclo - numCicloUltimaMejora >= 10)
            # println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
            criterioFin = true;
        end
    end


    # Devolvemos la RNA entrenada
    return (ann);
end;

# Función de crossValidation para CNN
function modelCrossValidationCNN(modelHyperparameters::Dict, inputs::Any, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})

    numFolds = modelHyperparameters["numFolds"];


    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);


    for numFold in 1:numFolds
        # Dividimos los datos en entrenamiento y test con los índices de crossValidation
        trainingInputs    = inputs[crossValidationIndices.!=numFold];
        testInputs        = inputs[crossValidationIndices.==numFold,:];
        trainingTargets   = targets[crossValidationIndices.!=numFold,:];
        testTargets       = targets[crossValidationIndices.==numFold,:];

        trainingInputs = convertirArrayImagenesHWCN(trainingInputs);
        testInputs     = convertirArrayImagenesHWCN(testInputs);


        testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
        testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);

        # Se entrena las veces que se haya indicado
        for numTraining in 1:modelHyperparameters["numExecutions"]

            ann = trainClassCNN((trainingInputs, trainingTargets), (testInputs, testTargets), modelHyperparameters["ann"]);

            (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = 
            confusionMatrix(vec(ann(testInputs)), vec(testTargets));
            
            #(testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix((ann(testInputs)'), testTargets);

        end;

        # Calculamos el valor promedio de todos los entrenamientos de este fold
        acc = mean(testAccuraciesEachRepetition);
        F1  = mean(testF1EachRepetition);

        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end; # for numFold in 1:numFolds

    println("CNN: Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println("CNN: Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));

end;



###### Ejecución

seed!(1)
numFolds = 10;
funcionTransferenciaCapasConvolucionales = relu;


dataset = loadDataset();
inputs  = dataset[1]; # RGB -> vector de matrices de 3D
targets = dataset[3];

indexVector = crossvalidation(targets, numFolds);


anns = [createCNN_simple1, createCNN_simple2, createCNN_simple3]; # El array tiene las funciones que crearán las RNA
execPerFold = 15;


println("\n\n\n###################################");
println("###        Deep Learning        ###")
println("###################################");

ruta = "resultados/aprox5_DL.csv";
cabecera = [("model", "meanPrecission", "stdPrecission", "meanF1", "stdF1")];
writedlm(ruta, cabecera, "&");
f = open(ruta, "a");


for i in eachindex(anns)
    print("\n\n**********************************\nCNN model number $(i).\n");
    modelHyperparametersCNN = Dict();
    modelHyperparametersCNN["ann"]= anns[i];
    modelHyperparametersCNN["numFolds"]= numFolds;
    modelHyperparametersCNN["numExecutions"] = execPerFold;

    
    fila = ["$i", round.(modelCrossValidationCNN(modelHyperparametersCNN, inputs, targets, indexVector), digits=4)...];
    writedlm(f, reshape(fila, (1,:)), "&");
end

close(f);
