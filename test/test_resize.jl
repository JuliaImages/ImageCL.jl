using ImageCL
using ImageCL.Resize

a = rand(UInt8, (128,128,16))

@time b = resize(a, (64,64,16); sampleAlgorithm="linear3D")
