using HDF5
using LinearAlgebra
using Plots

function knn(x, data, k)
    N = size(data)[1]
    V = zeros(0)
    newV = zeros(0)
    for i = 1:N
        val = norm(x - (data[i,1]))
        append!(V, val)
    end

    for j = 1:k
        append!(newV, V[j])
    end
    V = newV[k]

    return (k/(N*V))
end
function classifyKNN(x, data, h)
    x_y0 = zeros(0)
    x_y1 = zeros(0)
    zeroSum = 0
    oneSum = 0
    n = size(data)[1]
    for i = 1:n
        if (data[i,2] == 0.0)
            zeroSum+=1
        else (data[i,2] == 1.0)
            oneSum+=1
        end

        if (data[i,2] == 0.0)
            append!(x_y0, data[i])
        else
            append!(x_y1, data[i])
        end
    end
    p0 = knn(x,x_y0,h)
    p1 = knn(x,x_y1,h)

    zeroPercentage = zeroSum/n
    onePercentage = oneSum/n

    zerolikelihood = p0 * zeroPercentage
    onelikelihood = p1 * onePercentage

    if (zerolikelihood > onelikelihood)
        return 0
    end
    return 1
end

data = h5read("ps4.h5","data")

N = size(data)[1]
tempData = data
accuracySum = 0
error = 0
allErrors = zeros(0)
k = 1
bestError = 100.0
allK = zeros(0)
bestK = 0.0
yGuess = 0
i = 1
n = 1
l = 1
errorForK = 0

while k <= 30 #testing all h values
    global k
    global n
    while n <= 1000 #do each one 1000 times
        global tempData
        tempData = tempData[1:size(tempData,1) .!= i,: ]
        global l
        while l <= 999 #collecting sum of all accuracies for each point removed (run 999 times) skip 1 point at a time
            yGuess = classifyKNN(data[l,1],tempData,k)
            accuracySum += abs(yGuess - data[l,2])
            l+=1
        end
        tempData = data
        i+=1
        n+=1
        l=1
    end
    global accuracySum
    error = accuracySum/1000
    global allErrors
    error = error/1000
    append!(allErrors,error)
    append!(allK,k)
    global errorForK
    errorForK = 0
    accuracySum = 0
    n = 1
    global i
    i = 1
    k+=1
end

minErrorIndex = argmin(allErrors)
bestK = allK[minErrorIndex]

println(allErrors)
println(bestK)
plot(allK, allErrors)
