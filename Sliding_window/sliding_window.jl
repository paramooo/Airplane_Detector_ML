using JLD2
using Images
using Flux: onehotbatch, onecold, params
using JLD2, FileIO
using DelimitedFiles, Statistics, Random, Plots, Flux, Flux.Losses
using Statistics:mean
using PrettyTables
using ScikitLearn
using JLD2
using BSON: @load
using CUDA
using ThreadsX


function detect_planes(large_image, ann)
    # Define el tamaño de las subregiones y el tamaño del paso
    window_size = 20
    step_size = 5

    # Crea una copia de la imagen para dibujar los bordes
    output_image = copy(large_image)

    # Define the scales at which to extract windows
    scales = [0.7, 1.0, 1.3]

    # Calculate the total number of windows that will be extracted
    total_windows = 0
    for scale in scales
        # Calcula el tamaño deseado de la imagen redimensionada
        new_size = (round(Int, size(large_image, 1) * scale), round(Int, size(large_image, 2) * scale))

        # Redimensiona la imagen al tamaño deseado
        scaled_image = imresize(large_image, new_size)

        # Recalculate the number of windows for the current scale
        num_windows_x = div(size(scaled_image, 1) - window_size, step_size) + 1
        num_windows_y = div(size(scaled_image, 2) - window_size, step_size) + 1

        total_windows += num_windows_x * num_windows_y
    end

    println("Total de ventanas:", total_windows)
    # Preallocate an array for the windows
    windows = Array{Float32}(undef, window_size, window_size, size(large_image,3), total_windows)

    # Iterate over all scales
    window_index = 1
    for scale in scales
        # Calcula el tamaño deseado de la imagen redimensionada
        new_size = (round(Int, size(large_image, 1) * scale), round(Int, size(large_image, 2) * scale))

        # Redimensiona la imagen al tamaño deseado
        scaled_image = imresize(large_image, new_size)

        # Recalculate the number of windows for the current scale
        num_windows_x = div(size(scaled_image, 1) - window_size, step_size) + 1
        num_windows_y = div(size(scaled_image, 2) - window_size, step_size) + 1

        # Itera sobre todas las subregiones de la imagen más grande
        for i in 1:num_windows_x
            for j in 1:num_windows_y
                # Calcula las coordenadas de la subregión actual
                x = (i-1)*step_size + 1
                y = (j-1)*step_size + 1

                # Extrae una vista de la subregión actual y añádela al array windows
                windows[:,:,:,window_index] = float(@view scaled_image[x:x+window_size-1, y:y+window_size-1, :])
                window_index += 1
            end
        end
    end
    
    # Ejecuta la función en varias iteraciones
    group_size = 35000      #si esta con la GPU no se pueden poner numeros muy grandes porque se queda sin memoria, en cambio al no usarla da igual
    num_groups = ceil(Int, window_index / group_size)
    println("Número de grupos: $num_groups")
    
    #liberar memoria
    CUDA.reclaim()
    GC.gc()
    
    #pasar a la grafica los datos
    ann = gpu(ann)
    windows = gpu(windows)
    
    predictions = Vector{Array{Float32}}(undef, num_groups)
    for i in 1:num_groups
        predictions[i] = Array(ann(@view windows[:,:,:,(i-1)*group_size+1:min(i*group_size,end)]))
        println("Parte $i de $num_groups completada")
    end


    # Combina las predicciones en un solo array
    predictions = cat(predictions..., dims=2)

    println("Predicciones calculadas")
    # Itera sobre todas las subregiones de nuevo para dibujar los bordes si se detecta un avión en la subregión
    window_index = 1
    for scale in scales
        # Resize the image according to the current scale
        new_size = (round(Int, size(large_image, 1) * scale), round(Int, size(large_image, 2) * scale))
        scaled_image = imresize(large_image, new_size)
        
        # Recalculate the number of windows for the current scale
        num_windows_x = div(size(scaled_image, 1) - window_size, step_size) + 1
        num_windows_y = div(size(scaled_image, 2) - window_size, step_size) + 1

        for i in 1:num_windows_x
            for j in 1:num_windows_y
                # Calcula las coordenadas de la subregión actual en la imagen escalada
                x = (i-1)*step_size + 1
                y = (j-1)*step_size + 1

                # Si se detecta un avión en la subregión actual, dibuja un borde rojo alrededor de ella en la imagen original
                if predictions[1,window_index] >= 0.7
                    # Calcula las coordenadas de la subregión en la imagen original
                    x_orig = round(Int, x / scale)
                    y_orig = round(Int, y / scale)
                    window_size_orig = round(Int, window_size / scale)

                    if x_orig == 0
                        x_orig = 1
                    end
                    if y_orig == 0
                        y_orig = 1
                    end


                    #Dibuja el borde rojo                    
                    output_image[x_orig,y_orig:y_orig+window_size_orig-1,1] .= 1
                    output_image[x_orig,y_orig:y_orig+window_size_orig-1,2:3] .= 0

                    output_image[x_orig+window_size_orig-1,y_orig:y_orig+window_size_orig-1,1] .= 1
                    output_image[x_orig+window_size_orig-1,y_orig:y_orig+window_size_orig-1,2:3] .= 0

                    output_image[x_orig:x_orig+window_size_orig-1,y_orig,1] .= 1
                    output_image[x_orig:x_orig+window_size_orig-1,y_orig,2:3] .= 0

                    output_image[x_orig:x_orig+window_size_orig-1,y_orig+window_size_orig-1,1] .= 1
                    output_image[x_orig:x_orig+window_size_orig-1,y_orig+window_size_orig-1,2:3] .= 0

                end
                window_index += 1
            end
        end
    end
    println("Bordes dibujados") 
    # Devuelve la imagen con los bordes dibujados 
    return output_image 
end

# Carga la ANN desde el archivo 
#ann = load("finaldeslizante/ann3.jld2", "ann") 
@load "finaldeslizante/ann95.bson" ann

# Carga las imágenes de prueba y ejecuta la función detect_planes en cada una de ellas
for i in 1:4
    image = permutedims(channelview(load("finaldeslizante/test/test$i.png")), (3,2,1))
    if size(image, 3) == 4
        image = image[:,:,1:3]
    end
    println("Analizando imagen $i ...")
    local tiempo = @elapsed begin
        # Detecta los aviones en la imagen 
        local output_image = detect_planes(image, ann)
        save("finaldeslizante/outputs/output$i.png", output_image)
        println("Imagen guardada en finaldeslizante/outputs/output$i.png")
        CUDA.reclaim()
        GC.gc()
    end
    println("Tiempo total imagen $i: $tiempo\n")
end


