
using Flux: onehotbatch, onecold, params
using JLD2, FileIO
using DelimitedFiles, Statistics, Random, Plots, Flux, Flux.Losses
using Statistics:mean
using PrettyTables
using ScikitLearn
using JLD2
using BSON: @save
using CUDA

include("../fonts/load_images.jl")

function loadDataset2()
    (positivesColor, positivesGray) = loadFolderImages("finaldeslizante/planesnetpos");
    (negativesColor, negativesGray) = loadFolderImages("finaldeslizante/planesnetneg");
    targets = [trues(length(positivesColor)); falses(length(negativesColor))];
    return ([positivesColor; negativesColor], targets);
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

function trainClassCNN(train_set::Tuple{AbstractArray{<:Real,4}, AbstractArray{Bool,2}},
    test_set::Tuple{AbstractArray{<:Real,4}, AbstractArray{Bool,2}},
    generator::Function)
    (inputs, targets) = train_set
    (test_inputs, test_targets) = test_set

    # Comprobamos que el número de filas (número de patrones) coincide
    @assert(size(inputs, 4) == size(targets, 1))

    # Movemos los datos a la GPU
    inputs = gpu(inputs)
    targets = gpu(targets)
    test_inputs = gpu(test_inputs)
    test_targets = gpu(test_targets)
    
    println("Tamaño del conjunto de entrenamiento: ", size(inputs, 4))
    println("Tamaño del conjunto de test: ", size(test_inputs, 4))

    # Creamos la CNN y la movemos a la GPU
    ann = generator()
    ann = gpu(ann)

    # Definimos la función de accuracy
    accuracy(x, y) = mean((x .> 0.5) .== y)

    # Definimos la función de loss
    loss(x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y)

    opt = gpu(ADAM(0.001))
    mejorPrecision = -Inf
    criterioFin = false
    numCiclo = 0
    numCicloUltimaMejora = 0
    mejorModelo = nothing

    while !criterioFin
        numCiclo += 1

        Flux.train!(loss, params(ann), [(inputs, targets')], opt);

        precisionEntrenamiento = accuracy(ann(inputs), targets')

        println("Ciclo ", numCiclo, ": Precisión en el conjunto de entrenamiento: ", 100 * precisionEntrenamiento, " %")

        # Si se mejora la precisión en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
        if precisionEntrenamiento >= mejorPrecision
            mejorPrecision = precisionEntrenamiento
            precisionTest = accuracy(ann(test_inputs), test_targets')
            println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100 * precisionTest, " %")
            mejorModelo = deepcopy(cpu(ann))
            numCicloUltimaMejora = numCiclo
        end

        # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
        if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
            opt.eta /= 10.0
            println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta)
            numCicloUltimaMejora = numCiclo
        end

        # Criterios de parada:
        # Si la precisión en entrenamiento es lo suficientemente buena, se para el entrenamiento
        if precisionEntrenamiento >= 0.999
            println("   Se para el entrenamiento por haber llegado a una precisión de 99.9%")
            criterioFin = true
        end

        # Si no se mejora la precisión en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
        if numCiclo - numCicloUltimaMejora >= 10
            println("   Se para el entrenamiento por no haber mejorado la precisión en el conjunto de entrenamiento durante 10 ciclos")
            criterioFin = true
        end
    end

    # Devolvemos la RNA entrenada y la precisión en el conjunto de entrenamiento
    return mejorModelo, mejorPrecision
end


#Ejecucion
funcionTransferenciaCapasConvolucionales = relu;

dataset = loadDataset2();
inputs  = dataset[1]; # RGB -> vector de matrices de 3D
targets = dataset[2];

# Dividimos las imágenes positivas y negativas
println("Número de imágenes totales: ", length(inputs))
positive_imgs = inputs[1:sum(targets)]
negative_imgs = inputs[sum(targets)+1:end]

# Calculamos el número de imágenes para el conjunto de test
num_test_imgs_pos = Int(floor(0.03 * length(positive_imgs)))
num_test_imgs_neg = Int(floor(0.03 * length(negative_imgs)))

# Dividimos las imágenes positivas en conjuntos de entrenamiento y test
train_positive_imgs = positive_imgs[1:end-num_test_imgs_pos]
test_positive_imgs = positive_imgs[end-num_test_imgs_pos+1:end]

# Dividimos las imágenes negativas en conjuntos de entrenamiento y test
train_negative_imgs = negative_imgs[1:end-num_test_imgs_neg]
test_negative_imgs = negative_imgs[end-num_test_imgs_neg+1:end]

println("Número de imágenes positivas para entrenamiento: ", length(train_positive_imgs))
println("Número de imágenes positivas para test: ", length(test_positive_imgs))

println("Número de imágenes negativas para entrenamiento: ", length(train_negative_imgs))
println("Número de imágenes negativas para test: ", length(test_negative_imgs))

# Combinamos las imágenes positivas y negativas para formar los conjuntos de entrenamiento y test
train_imgs = [train_positive_imgs; train_negative_imgs]
test_imgs = [test_positive_imgs; test_negative_imgs]

# Creamos los targets para los conjuntos de entrenamiento y test
trainingTargets = BitMatrix([trues(length(train_positive_imgs)); falses(length(train_negative_imgs))][:,:])
testTargets = BitMatrix([trues(length(test_positive_imgs)); falses(length(test_negative_imgs))][:,:])


# Convertimos los datos a un formato adecuado para la red neuronal
trainingInputs = convertirArrayImagenesHWCN(train_imgs)
testInputs     = convertirArrayImagenesHWCN(test_imgs)

println("Número de imagenes para entrenamiento: ", length(trainingTargets))
println("Número de imagenes para test: ", length(testTargets))

println("\n\n###########################################");
println("###        Deep Learning -> Model 2   ###")
println("###########################################");

# Entrenamos la red neuronal
precision = 0.0
global precision = 0.0
global mejorprecision = 0.0
while precision < 0.95
    global ann, precision = trainClassCNN((trainingInputs, trainingTargets), (testInputs, testTargets), createCNN_2);
    CUDA.reclaim()
    GC.gc()
    println("Precisión en el conjunto de entrenamiento del modelo: ", 100 * precision, " %")
    if precision>0.92 && precision>mejorprecision
        global mejorprecision = precision
        @save "finaldeslizante/ann92temp.bson" ann
        println("Modelo temporal guardado en ann92temp.bson con precision ", 100 * precision, " %")
    end
end

@save "finaldeslizante/ann.bson" ann
println("Modelo guardado en ann.bson")



