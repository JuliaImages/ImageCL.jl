module ImageCL
using OpenCL

# export device, ctx, queue, load_kernel
export load_kernel
# package code goes here

# device, ctx, queue = OpenCL.cl.create_compute_context()

load_kernel(name) = open(readstring, joinpath(dirname(@__FILE__), "..", "kernel", name))

#include("orb.jl")
# include("gaussian_blur.jl")
include("resize.jl")
include("gaussian_blur.jl")
end # module
