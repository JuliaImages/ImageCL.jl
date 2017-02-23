module ImageCL

export load_kernel
# package code goes here

load_kernel(name) = open(readall, joinpath(dirname(@__FILE__), "..", "kernel", name))

#include("orb.jl")
# include("gaussian_blur.jl")

end # module
