module Resize

using OpenCL
import  OpenCL: cl
# using Colors, FileIO, Images
using ..ImageCL
# using ImageView

export resize, resize!

const RESIZE_KERNEL_SOURCE = ImageCL.load_kernel( "resize.cl" )

"""
    resize{T,N}( inArray::Array{T,N}, outSize::NTuple{N,Int}; sampleAlgorithm = "linear3D" )
change the value inplace of outArray.
sampleAlgorithm = "linear3D" | "linear2D" | "nearest2D" | "nearest3D"
"""
function resize{T,N}( inArray::Array{T,N}, outSize::NTuple{N,Int};
                        sampleAlgorithm = "linear3D")
    device, ctx, queue = OpenCL.cl.create_compute_context()
    resize_program = cl.Program(ctx, source=RESIZE_KERNEL_SOURCE) |> cl.build!
    resize_cl_func = cl.Kernel(resize_program, sampleAlgorithm)
    inBuffer    = cl.Buffer(eltype(inArray), ctx, (:r, :copy), hostbuf = inArray)
    outBuffer   = cl.Buffer(eltype(inArray), ctx, :w,          prod(outSize))
    # the group size should be the maximum of in and out array size
    # since we are doing downsampling, so the maximum size is the input size
    # https://github.com/seung-lab/opencl-imageproc/blob/master/resize.py#L202
    resize_cl_func[queue,size(inArray)](inBuffer, outBuffer)
    ret = reshape(cl.read(queue, outBuffer), outSize)
end

end # end of module
