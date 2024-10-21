using DelimitedFiles, Statistics, Random, Plots, Flux, Flux.Losses
using Random:seed!
using Flux: params
using PrettyTables
using ScikitLearn

@sk_import svm: SVC 
@sk_import tree: DecisionTreeClassifier 
@sk_import neighbors: KNeighborsClassifier


#feature -> vector con los valores de un atributo o salida deseada para cada patron
#classes -> valores de las categorias
function oneHotEncoding(feature::AbstractArray{<:Any,1},classes::AbstractArray{<:Any,1})
    # Comprobamos que todos los elementos del vector aparecen en classes
    @assert(all([in(value, classes) for value in feature]));

    if(length(classes) == 2)
        rtn = reshape(feature .== classes[1], :, 1);
    else
        rtn = BitArray{2}(undef, length(feature), length(classes));
        for i in classes
            rtn[:,i] .= (feature .== classes[i]);
        end
    end
    return rtn;  
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    minimo = minimum(dataset, dims=1);
    maximo = maximum(dataset, dims=1);
    return (minimo,maximo);
end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    media = mean(dataset, dims=1);
    desviacion_tipica = std(dataset, dims=1);
    return (media,desviacion_tipica);
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    num_columns = length(normalizationParameters);
    for i in 1:num_columns
        if normalizationParameters[1][i] == normalizationParameters[2][1]
            dataset[:,i] .= 0;
        else
            dataset .-= normalizationParameters[1];
            dataset ./= (normalizationParameters[2] - normalizationParameters[1]);
        end
    end
end

normalizeMinMax!(dataset::AbstractArray{<:Real,2}) = normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset));

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    copied = copy(dataset);
    normalizeMinMax!(copied, normalizationParameters);
    return copied;
end

normalizeMinMax(dataset::AbstractArray{<:Real,2}) = normalizeMinMax(dataset, calculateMinMaxNormalizationParameters(dataset));


function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    num_columns = length(normalizationParameters);
    for i in 1:num_columns
        if normalizationParameters[2][i] == 0
            dataset[:,i] .= 0;
        else
            dataset[:,i] = (dataset[:,i] .- normalizationParameters[1][i]) ./ normalizationParameters[2][i];
        end
    end
end

normalizeZeroMean!(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset));

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    copied = copy(dataset);
    normalizeZeroMean!(copied, normalizationParameters);
    return copied;
end

normalizeZeroMean(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean(dataset, calculateZeroMeanNormalizationParameters(dataset));

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    num_columns = size(outputs,2);
    if(num_columns == 1)
        outputs = (outputs .>= threshold);
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
    end
    return outputs;
end


accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean((outputs .== targets));

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    if((size(outputs,2) == 1) && (size(targets,2) == 1))
        accuracy(outputs[:,1], targets[:,1]);
    elseif((size(outputs,2) > 2) && (size(targets,2) > 2))
        classComparison = targets .== outputs;
        correctClassifications = all(classComparison, dims=2);
        return mean(correctClassifications);
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) 
    outputs = (outputs .>= threshold);
    accuracy(outputs,targets);
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5) 
    outputs = (outputs .>= threshold);
    if((size(outputs,2) == 1) && (size(targets,2) == 1))
        accuracy(targets[:, 1], outputs[:, 1]);
    elseif((size(outputs,2) > 2) && (size(targets,2) > 2)) 
        outputs = classifyOutputs(outputs);
        accuracy(targets, outputs);
    end
end


function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 

    ann = Chain();
    numInputsLayer = numInputs;
    iteration = 1;
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer,  transferFunctions[iteration]));
        numInputsLayer = numOutputsLayer;
        iteration += 1;
    end
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end
    return ann;
end

#Funcion para dividir el dataset en 2 subconjuntos: entranamiento y test
function holdOut(N::Int, P::Real)
    @assert ((P>=0.) & (P<=1.));
    #Vector permutado de tamaño N
    randomVector = randperm(N);

    numTrainingInstances = Int(round(N*(1-P)));
    return (randomVector[1:numTrainingInstances], randomVector[numTrainingInstances+1:end]);
end

#Funcion para dividir el dataset en 3 subconjuntos: entranamiento, validación y test
function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest )<= 1.);

    # Separamos en entrenamiento/validacion y test
    (trainingValidationIndices, testIndices) = holdOut(N, Ptest);
    # Separamos el conjunto de entrenamiento y validacion
    (trainingIndices, validationIndices) = holdOut(length(trainingValidationIndices), Pval*N/length(trainingValidationIndices))
    return (trainingValidationIndices[trainingIndices], trainingValidationIndices[validationIndices], testIndices);
end



function trainClassANN(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    testDataset:: Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
    maxEpochsVal::Int=20, showText::Bool=false)

    #Separamos las entradas y salidas deseadas de los conjuntos de entrenamiento, validacion y test y
    #comprobamos que hay el mismo número de patrones(filas) en las entradas y en las salidas deseadas
    trainingInputs = trainingDataset[1];
    trainingTargets = trainingDataset[2]; 
    @assert(size(trainingInputs,1)==size(trainingTargets,1));

    if(validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && validationDataset[2] != falses(0,0))
        validationInputs = validationDataset[1];
        validationTargets = validationDataset[2];
        @assert(size(validationInputs,1)==size(validationTargets,1));
    end

    if(testDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && testDataset[2] != falses(0,0))
        testInputs = testDataset[1];
        testTargets = testDataset[2];
        @assert(size(testInputs,1)==size(testTargets,1));
    end
    

    #Creamos la RNA
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions=transferFunctions);

    #Creamos la funcion "loss" para entrenar la RNA
    #Dependiendo de si hay 2 clases o más de 2 usamos una función u otra
    #El primer argumento son las salidas del modelo y el segundo las salidas deseadas
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    #Vectores con los valores de loss en cada ciclo de entrenamiento
    lossTraining = Float32[];
    lossValidation = Float32[];
    lossTest = Float32[];

    #Obtenemos los valores de loss en el ciclo 0 (los pesos son aleatorios) y los almacenamos
    lossTrainingCurrent = loss(trainingInputs', trainingTargets');
    push!(lossTraining, lossTrainingCurrent);

    if(!isempty(validationInputs))
        lossValidationCurrent = loss(validationInputs', validationTargets');
        push!(lossValidation, lossValidationCurrent);
    else
        lossValidationCurrent = NaN;
    end

    if(!isempty(testInputs))
        lossTestCurrent = loss(testInputs', testTargets');
        push!(lossTest, lossTestCurrent);
    else
        lossTestCurrent = NaN;
    end
    

    #Ciclo actual, nº de ciclos sin mejorar el loss de validación, mejor error de valoración, mejor RNA
    currentEpoch = 0;
    epochNoUpgradeValidation = 0;
    if(validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && validationDataset[2] != falses(0,0))
        bestValidationLoss = lossValidationCurrent;
    end
    bestANN = deepcopy(ann);
    
    while ((currentEpoch < maxEpochs) && (lossTrainingCurrent > minLoss) && (epochNoUpgradeValidation < maxEpochsVal))

        #Entrenamos un ciclo la RNA
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));

        #Aumentamos el ciclo actual
        currentEpoch += 1;

        #Obtenemos los valores de loss en el ciclo actual y los almacenamos
        lossTrainingCurrent = loss(trainingInputs', trainingTargets');
        push!(lossTraining, lossTrainingCurrent);

        if(!isempty(validationInputs))
            lossValidationCurrent = loss(validationInputs', validationTargets');
            push!(lossValidation, lossValidationCurrent);

            #Si mejoramos el error, guardamos la RNA y ponemos a 0 el nº de ciclos sin mejora (Parada temprana)
            if(lossValidationCurrent < bestValidationLoss)
                bestValidationLoss = lossValidationCurrent;
                epochNoUpgradeValidation = 0;
                bestANN = deepcopy(ann);
            else
                epochNoUpgradeValidation += 1;
            end    

        end  

        if(testDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,0) && testDataset[2] != falses(0,0))
            lossTestCurrent = loss(testInputs', testTargets');
            push!(lossTest, lossTestCurrent);
        end          
            
    end

    if(!isempty(validationInputs))
        return (bestANN, lossTraining, lossValidation, lossTest);
    else
        return (ann, lossTraining, lossValidation, lossTest);
    end
    
end


function trainClassANN(topology::AbstractArray{<:Int,1}, 
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
    (Array{eltype(trainingDataset[1]),1}(undef,0,0), falses(0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
    (Array{eltype(trainingDataset[1]),1}(undef,0,0), falses(0)), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
    maxEpochsVal::Int=20, showText::Bool=false) 


    trainClassANN(topology, (trainingDataset[1], reshape(trainingDataset[2], 1)), 
    validationDataset = (validationDataset[1], reshape(validationDataset[2], 1)), 
    testDataset = (testDataset[1], reshape(testDataset[2], 1)),
    transferFunctions = transferFunctions, maxEpochs = maxEpochs, minLoss = minLoss, 
    learningRate = learningRate, maxEpochsVal = maxEpochsVal, showText = showText);

end


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #Comprobamos que los vectores de salidas obtenidas y salidas deseadas sean de la misma longitud
    @assert(length(outputs)==length(targets));

    #Obtenemos los valores de VP, VN, FP, FN
    vp = sum(targets .& outputs);
    vn = sum(.!targets .& .!outputs);
    fp = sum(.!targets .& outputs);
    fn = sum(targets .& .!outputs);


    #Obtenemos la precisión y la tasa de error utilizando las funciones auxiliares
    acc = accuracy(outputs,targets);
    errorRate = 1. - acc;

    #Calculamos la sensibilidad, la especificidad, el valor predictivo positivo, el valor predictivo negativo y la F1-score
    recall = vp / (fn + vp);
    specificity = vn / (fp + vn);
    ppv = vp / (vp + fp);
    npv = vn / (vn + fn)
    f1 = (2 * recall * ppv) / (recall + ppv); 

    #Calculamos la matriz de confusión
    conf_matrix = Array{Int64,2}(undef, 2, 2);
    conf_matrix[1,1] = vn;
    conf_matrix[1,2] = fp;
    conf_matrix[2,1] = fn;
    conf_matrix[2,2] = vp;

    #Tenemos en cuenta varios casos particulares
    if (vn == length(targets))
        recall = 1.;
        ppv = 1.;
    elseif (vp == length(targets))
        specificity = 1.;
        npv = 1.;
    end

    recall = isnan(recall) ? 0. : recall;
    specificity = isnan(specificity) ? 0. : specificity;
    ppv = isnan(ppv) ? 0. : ppv;
    npv = isnan(npv) ? 0. : npv;

    f1 = (recall == ppv == 0.) ? 0. : 2 * (recall * ppv) / (recall + ppv);

    return (acc, errorRate, recall, specificity, ppv, npv, f1, conf_matrix);

end


function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) 
    confusionMatrix(AbstractArray{Bool,1}(outputs.>=threshold),targets);
end


function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    (acc, errorRate, recall, specificity, ppv, npv, f1, conf_matrix) = confusionMatrix(outputs,targets);

    #Mostramos los datos por pantalla
    print("Valor de precisión: ", acc, "\n");
    print("Tasa de fallo: ", errorRate, "\n");
    print("Sensibilidad: ", recall, "\n");
    print("Especificidad: ", specificity, "\n");
    print("Valor predictivo positivo: ", ppv, "\n");
    print("Valor predictivo negativo: ", npv, "\n");
    print("F1-Score: ", f1, "\n");
    
    #Dibujamos la matriz
    print("Matriz de confusión: \n");

    rows = ["Real Negativo", "Real Positivo"];
    columns = ["Predicción Negativo", "Predicción Positivo"];

    pretty_table(conf_matrix; header=columns, row_names=rows);
end


function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    printConfusionMatrix(AbstractArray{Bool,1}(outputs.>=threshold),targets);    
end


#Estrategia "Uno contra todos"
function oneVSall(model, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}) 
    #Obtenemos el número de clases y de instancias
    numClasses = size(targets,2);
    numInstances = length(inputs);

    #Comprobamos que el número de clases sea mayor que 2
    @assert(numClasses>2);

    #Creamos una matriz bidimensional con tantas filas como patrones y tantas columnas como clases 
    outputs = Array{Float32,2}(undef, numInstances, numClasses);

    #Realizamos un bucle que itere sobre cada clase. 
    #Creamos las salidas deseadas a cada clase y se entrena el modelo
    for numClasses in 1:numClasses
        newModel = deepcopy(model);
        fit!(newModel, inputs, targets[:, [numClasses]]);
        outputs[:,numClasses] .= newModel(inputs);
    end

    #Tomamos la salida mayor de cada clase, aplicándole antes la funcion softmax
    outputs = softmax(outputs')';
    outputs = classifyOutputs(outputs);
    classComparison = (targets .== outputs);
    correctClassifications = all(classComparison, dims=2);
    return mean(correctClassifications);
end

#Métricas para el caso de tener más de dos clases
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) 
    #Comprobamos que los números de columnas de outputs y targets son iguales y distintos de 2
    @assert(size(outputs,2) == size(targets,2));
    numClasses = size(targets,2);
    @assert(numClasses != 2);

    #Si el nº de columnas es igual a 1 llamamos a la función anterior
    if(numClasses == 1)
        return confusionMatrix(outputs[:,1],targets[:,1]);
    else
        #Reservamos memoria para los vectores de las métricas, con un valor por clase
        recall = zeros(numClasses);
        specificity = zeros(numClasses);
        ppv = zeros(numClasses);
        npv = zeros(numClasses);
        f1 = zeros(numClasses); 

        #Iteramos para cada clase, obteniendo las métricas 
        patternsEachClass = vec(sum(targets, dims=1));
        for numClass in 1:numClasses
            if (patternsEachClass[numClass] != 0)
                (_, _, recall[numClass], specificity[numClass], ppv[numClass], npv[numClass], f1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);              
            end
        end

        #Reservamos memoria para la matriz de confusión
        conf_matrix = Array{Int64,2}(undef, numClasses, numClasses);

        #Realizamos un bucle doble para rellenar la matriz de confusión
        for numClassTarget in 1:numClasses
            for numClassOutput in 1:numClasses
                conf_matrix[numClassTarget,numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
            end
        end

        #Tomamos los valores dependiendo de si es macro o weighted
        if (weighted)
            weight = patternsEachClass ./ sum(patternsEachClass);
 
            recall = sum(weight .* recall);
            specificity = sum(weight .* specificity);
            ppv = sum(weight .* ppv);
            npv = sum(weight .* npv);
            f1 = sum(weight .* f1);

        else
            #Para hacer la media solo usamos las clases que tienen patrones
            nonZeroClasses = sum(patternsEachClass .> 0);

            recall = sum(recall)/nonZeroClasses;
            specificity = sum(specificity)/nonZeroClasses;
            ppv = sum(ppv)/nonZeroClasses;
            npv = sum(npv)/nonZeroClasses;
            f1 = sum(f1)/nonZeroClasses;

        end
        #Calculamos la precisión y la tasa de error
        acc = accuracy(outputs, targets);
        errorRate = 1. - acc;

        return(acc, errorRate, recall, specificity, ppv, npv, f1, conf_matrix);
    end   
end


function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) 
    return confusionMatrix(classifyOutputs(outputs), targets, weighted = weighted);
end


function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(output, unique(targets)) for output in outputs]))
    classes = unique(targets);
    confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end


function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, ppv, npv, f1, conf_matrix) = confusionMatrix(outputs,targets, weighted = weighted);

    #Mostramos los datos por pantalla
    print("Valor de precisión: ", acc, "\n");
    print("Tasa de fallo: ", errorRate, "\n");
    print("Sensibilidad: ", recall, "\n");
    print("Especificidad: ", specificity, "\n");
    print("Valor predictivo positivo: ", ppv, "\n");
    print("Valor predictivo negativo: ", npv, "\n");
    print("F1-Score: ", f1, "\n");

    #Dibujamos la matriz
    print("Matriz de confusión: \n");

    rows = String["Clase " * string(i) for i in 1:size(conf_matrix, 1)];
    columns = String["Clase " * string(i) for i in 1:size(conf_matrix, 2)];

    pretty_table(conf_matrix; header=columns, row_names=rows);

end


function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);
end



function crossvalidation(N::Int64, k::Int64) 
    nVector = repeat(1:k, Int64(ceil(N/k)));
    #Tomamos los N primeros valores
    nVector = nVector[1:N];
    #Desordenamos el vector y lo devolvemos
    shuffle!(nVector);
    return nVector;
end


function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64) 
    numClasses = size(targets,2);
    indexes = Array{Int64,1}(undef, size(targets,1));
    for numClass in 1:numClasses
        #Comprobamos que haya al menos k patrones de cada clase
        @assert(sum(targets[:,i]) >= k);

        indexes[targets[:,numClass]] = crossvalidation(sum(targets[:,numClass]), k);
    end
    return indexes;
end


function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    classes = unique(targets);
    indexes = Array{Int64,1}(undef, length(targets));
    for class in classes
        indicesThisClass = (targets .== class);
        indexes[indicesThisClass] = crossvalidation(sum(indicesThisClass), k);
    end;
    return indexes;
end 


function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
    kFoldIndices:: Array{Int64,1}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, 
    minLoss::Real=0.0, learningRate::Real=0.01, numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0, maxEpochsVal::Int=20) 

    #Calculamos el número de folds
    numfolds = maximum(kFoldIndices);

    #Creamos un vector que almacene los valores de precisión
    accVector = Array{Float64,1}(undef, numfolds);
    f1Vector = Array{Float64,1}(undef, numfolds);

    #Obtenemos los inputs y los targets del dataset
    inputs = trainingDataset[1];
    targets = trainingDataset[2];

    #Bucle con k iteraciones (k = numfolds)
    for i in 1:numfolds
        #Creamos las matrices de entradas y salidas deseadas de entrenamiento y test
        trainInputs = inputs[kFoldIndices .!= i,:]
        testInputs = inputs[kFoldIndices .== i, :]
        trainTargets = targets[kFoldIndices .!= i,:]
        testTargets = targets[kFoldIndices .== i,:]

        #El entrenamiento de RNA no es determinístico, por lo que, para cada iteración k de la validación cruzada, 
        #será necesario entrenar varias RNA y devolver el promedio de los resultados de test

        #Creamos un vector para almacenar la metrica en cada repeticion
        accPerRep = Array{Float64,1}(undef, numRepetitionsANNTraining);
        f1PerRep = Array{Float64,1}(undef, numRepetitionsANNTraining);
        for j in 1:numRepetitionsANNTraining
            #Comprobamos si vamos a emplear conjunto de validación
            if (validationRatio > 0)
                #Dividimos el conjunto de entrenamiento en entrenamiento y validación
                #Entendemos que el ratio de validacion es sobre el total de patrones, contando los de test
                (trainIndexes, validationIndexes) = holdOut(size(trainInputs,1), validationRatio*size(trainInputs,1)/size(inputs,1));

                #Entrenamos la RNA
                ann, trainingLosses, testLosses, validationLosses = trainClassANN(topology, (convert(Array{Real,2}, trainInputs[trainIndexes,:]), convert(Array{Bool,2}, trainTargets[trainIndexes,:])), 
                                    validationDataset = (convert(Array{Real,2}, trainInputs[validationIndexes,:]), convert(Array{Bool,2}, trainTargets[validationIndexes,:])),
                                    testDataset = (testInputs, testTargets), transferFunctions = transferFunctions,
                                    maxEpochs = maxEpochs, learningRate = learningRate, minLoss = minLoss, maxEpochsVal = maxEpochsVal);


                #Mostramos la gráfica de loss de la primera iteracón del primer fold
                if (j==1 && i == 1)                      
                    g = plot(title = "Evolución de los valores de loss", xaxis = "Epoch", yaxis = "MSE")
                    plot!(g,1:length(trainingLosses),trainingLosses,label="Error de entrenamiento",color="green")
                    plot!(g,1:length(testLosses),testLosses,label="Error de test",color="red")
                    plot!(g,1:length(validationLosses),validationLosses,label="Error de validacion",color="blue")   
                    display(g) 
                end;

            else
                #Con la "," obtenemos solo el primer valor (la red neuronal)
                ann, = trainClassANN(topology, (trainInputs, trainTargets), testDataset = (testInputs, testTargets), 
                                    transferFunctions = transferFunctions, maxEpochs = maxEpochs, 
                                    learningRate = learningRate, minLoss = minLoss);
            end
            #Obtenemos las métricas que queremos y las almacenamos en los vectores correspondientes
            (acc, _, _, _, _, _, f1, _) = confusionMatrix(collect(ann(testInputs')'), testTargets);
            accPerRep[j] = acc;
            f1PerRep[j] = f1;            
        end
        #Almacenamos las métricas que queremos y las mostramos por pantalla
        accVector[i] = mean(accPerRep);
        f1Vector[i] = mean(f1PerRep);
        println("ANN Test ACCURACY results for fold ", i, "/", numfolds, ": ", accVector[i]);
    end
    #Mostramos por pantalla la media de las métricas deseadas y las devolvemos
    println("ANN Average test accuracy (", numfolds, " folds): ", mean(accVector), ", std desviation: ", std(accVector));
    println("ANN Average test f1-score (", numfolds, " folds): ", mean(f1Vector), ", std desviation: ", std(f1Vector));

    return (mean(accVector), std(accVector));
end

#= 
function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, 
    targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1}) 

    #Comprobamos que el número de patrones coincida
    @assert(size(inputs,1)==length(targets));
    #Obtenemos las clases 
    classes = unique(targets);

    if (modelType == :ANN)
        #Hacemos one-hot-encoding en el caso de que vayamos a emplear una RNA
        targets = oneHotEncoding(targets,classes);

        #Creamos y entrenamos la RNA
        return trainClassANN(modelHyperparameters["topology"], (inputs,targets), crossValidationIndices, 
                                    maxEpochs=modelHyperparameters["maxEpochs"], 
                                    minLoss=modelHyperparameters["minLoss"], 
                                    learningRate=modelHyperparameters["learningRate"],
                                    numRepetitionsANNTraining=modelHyperparameters["numRepetitionsANNTraining"],
                                    validationRatio=modelHyperparameters["validationRatio"],
                                    maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

    elseif ((modelType == :SVM) || (modelType == :DecisionTree) || (modelType == :kNN))
        #Calculamos el número de folds
        numfolds = maximum(crossValidationIndices);

        #Creamos un vector que almacene los valores de precisión
        accVector = Array{Float64,1}(undef, numfolds);
        f1Vector = Array{Float64,1}(undef, numfolds);

        #Arrays generales para almacenar los resultados de las predicciones
        testOutputsGeneral = Float64[];
        testTargetsGeneral = Float64[];

        #Bucle con k iteraciones (k = numfolds)
        for i in 1:numfolds
            #Creamos las matrices de entradas y salidas deseadas de entrenamiento y test
            trainingInputs = inputs[crossValidationIndices .!= i,:];
            testInputs = inputs[crossValidationIndices .== i, :];
            trainingTargets = targets[crossValidationIndices .!= i];
            testTargets = targets[crossValidationIndices .== i];

            #Generamos el modelo correspondiente
            if (modelType == :SVM)
                model = SVC(kernel=modelHyperparameters["kernel"],
                        degree=modelHyperparameters["degreeKernel"],
                        gamma=modelHyperparameters["gammaKernel"], 
                        C=modelHyperparameters["C"]);
            elseif (modelType == :DecisionTree)
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            else
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end

            #Entrenamos el modelo correspondiente con el conjunto de entranamiento
            model = fit!(model, trainingInputs, trainingTargets);

            #Realizamos predicciones con el modelo entrenado
            testOutputs = predict(model, testInputs);
        
            #Calculamos las métricas deseadas
            (acc, _, _, _, _, _, f1, _) = confusionMatrix(testOutputs, testTargets);
            testOutputsGeneral = [testOutputsGeneral;  testOutputs];
            testTargetsGeneral = [testTargetsGeneral; testTargets];

            accVector[i] = acc;
            f1Vector[i] = f1;
            println(modelType, " Test ACCURACY results for fold ", i, "/", numfolds, ": ", accVector[i]);
        end
 
        #Mostramos por pantalla la media de las métricas deseadas y las devolvemos
        println(modelType, " Average test accuracy (", numfolds, " folds): ", mean(accVector), ", std desviation: ", std(accVector));
        println(modelType, " Average test f1-score (", numfolds, " folds): ", mean(f1Vector), ", std desviation: ", std(f1Vector));

        #Imprimimos la matriz de confusión
        printConfusionMatrix(testOutputsGeneral.==1, testTargetsGeneral.==1)
        return (mean(accVector), std(accVector));
    end
end
 =#

 
function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Que clases de salida tenemos
    # Es importante calcular esto primero porque se va a realizar codificacion one-hot-encoding varias veces, y el orden de las clases deberia ser el mismo siempre
    classes = unique(targets);

    # Primero codificamos las salidas deseadas en caso de entrenar RR.NN.AA.
    if modelType==:ANN
        targets = oneHotEncoding(targets, classes);
    end;

    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Si vamos a usar unos de estos 3 modelos
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold];
            testTargets       = targets[crossValidationIndices.==numFold];

            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            # Entrenamos el modelo con el conjunto de entrenamiento
            model = fit!(model, trainingInputs, trainingTargets);

            # Pasamos el conjunto de test
            testOutputs = predict(model, testInputs);

            # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);

        else

            # Vamos a usar RR.NN.AA.
            @assert(modelType==:ANN);

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold,:];
            testTargets       = targets[crossValidationIndices.==numFold,:];

            # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
            #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento
            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);

            # Se entrena las veces que se haya indicado
            for numTraining in 1:modelHyperparameters["numExecutions"]

                if modelHyperparameters["validationRatio"]>0

                    # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                    #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                    #  Para ello, hacemos un hold out
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

                    # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                    ann,= trainClassANN(modelHyperparameters["topology"], (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                        validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                        testDataset =       (testInputs,                          testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

                    #= ann, trainingLosses, testLosses, validationLosses = trainClassANN(modelHyperparameters["topology"], (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                        validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                        testDataset =       (testInputs,                          testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

                    #Mostramos la gráfica de loss de la primera iteracón del primer fold
                    if (numFold==1 && numTraining == 1)                      
                        g = plot(title = "Evolución de los valores de loss", xaxis = "Epoch", yaxis = "MSE")
                        plot!(g,1:length(trainingLosses),trainingLosses,label="Error de entrenamiento",color="green")
                        plot!(g,1:length(testLosses),testLosses,label="Error de test",color="red")
                        plot!(g,1:length(validationLosses),validationLosses,label="Error de validacion",color="blue")   
                        display(g) 
                    end =#

                else

                    # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                    #  teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"], (trainingInputs, trainingTargets),
                        testDataset = (testInputs,     testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"]);

                end;

                # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            end;

            # Calculamos el valor promedio de todos los entrenamientos de este fold
            acc = mean(testAccuraciesEachRepetition);
            F1  = mean(testF1EachRepetition);

        end;

        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end; # for numFold in 1:numFolds

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));

end;




numFolds = 10;