module GaussianBlur

using OpenCL
import  OpenCL: cl
# using Colors, FileIO, Images
using ..ImageCL
using Images
# using ImageView

export blur!, blur

const GAUSS_KERNEL_SOURCE = ImageCL.load_kernel( "gaussian_blur.cl" )

"""
    blur!{T,N}( inArray::Array{T,N}, outArray::Array{T,N}; samplingAlg = "linear_3d" )
change the value inplace of outArray.
blurAlg = "fir_ver_blur"
"""
function blur!{T,N}( inArray::Array{T,N}, kernel::Array;
                        blurAlg = "fir_ver_blur" )
    device, ctx, queue = OpenCL.cl.create_compute_context()
    gauss_program = cl.Program(ctx, source=GAUSS_KERNEL_SOURCE) |> cl.build!
    gauss_cl_func = cl.Kernel(gauss_program, blurAlg)
    @show gauss_cl_func

    inBuffer    = cl.Buffer(RGBA{Float32},  ctx, (:r, :copy),   hostbuf = inArray)
    outBuffer   = cl.Buffer(RGBA{Float32},  ctx, :w,            length(inArray))
    kernelBuffer= cl.Buffer(Float32,        ctx, (:r, :copy),   hostbuf=kernel)
    gauss_cl_func[queue,size(inArray)](
                            inBuffer,       Cint(size(inArray, 1)),
                            outBuffer,
                            kernelBuffer,   Cint(length(kernel)),
                            Cint(0));
    ret = reshape(cl.read(queue, outBuffer), size(inArray))
    return ret
end

function blur{T,N}( inArray::Array{T,N};
                    kernelSigmas::Tuple=(2.0, 2.0),
                    kernelSize::Tuple=(5,5),
                    blurAlg = "fir_ver_blur")
    outArray = similar(inArray)
    kernel = map(Float32, Images.Kernel.gaussian(kernelSigmas, kernelSize)).parent
    blur!(inArray, kernel, outArray; blurAlg=blurAlg)
end

end # end of module
