using HDF5
using LinearAlgebra
using Plots

function kernel(x, data, h)
    N = size(data)[1]
    psum = 0
    for i = 1:N
        firstpart = 1/sqrt(2*pi*h^2)
        secondpart = exp(-norm(x-data[i])^2/(2*h^2))
        psum += firstpart*secondpart
    end
    return p = psum/N
end

function classifyParzen(x, data, h)
    x_y0 = zeros(0)
    x_y1 = zeros(0)
    zeroSum = 0
    oneSum = 0
    n = size(data)[1]
    for i = 1:n
        if (data[i,2] == 0.0)
            zeroSum+=1
        else
            oneSum+=1
        end

        if (data[i,2] == 0.0)
            append!(x_y0, data[i])
        else
            append!(x_y1, data[i])
        end
    end
    p0 = kernel(x,x_y0,h)
    p1 = kernel(x,x_y1,h)

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
h = 0.1
bestError = 100.0
allH = zeros(0)
bestH = 0.0
yGuess = 0
i = 1
n = 1
l = 1
errorForH = 0
while h <= 2 #testing all h values
    global h
    global n
    while n <= 1000 #do each one 1000 times
        global tempData
        tempData = tempData[1:size(tempData,1) .!= i,: ]
        global l
        while l <= 999 #collecting sum of all accuracies for each point removed (run 999 times) skip 1 point at a time
            yGuess = classifyParzen(data[l,1],tempData,h)
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
    append!(allErrors,error)
    append!(allH,h)
    global errorForH
    errorForH = 0
    accuracySum = 0
    n = 1
    global i
    i = 1
    h+=0.1
end

minErrorIndex = argmin(allErrors)
bestH = allH[minErrorIndex]

println(bestH)
plot(allH,allErrors)
