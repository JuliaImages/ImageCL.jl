using OpenCL
import  OpenCL: cl
using Colors, FileIO, Images
using ImageCL
using Images
using ImageView

image = map(Colors.RGBA{Float32}, load(joinpath(dirname(@__FILE__()), "../assets/cups.jpg")))
imshow(image)

device, ctx, queue = OpenCL.cl.create_compute_context()
const gauss_kernel_src = ImageCL.load_kernel("gaussian_blur.cl")
gauss_program = cl.Program(ctx, source=gauss_kernel_src) |> cl.build!
gauss_cl_func = cl.Kernel(gauss_program, "fir_ver_blur")

# create opencl buffer objects
# copies to the device initiated when the kernel function is called
kernel = map(Float32, Images.Kernel.gaussian((2.0,2.0), (5,5))).parent
img_buff     = cl.Buffer(RGBA{Float32}, ctx, (:r, :copy),   hostbuf=image)
blurred_buff = cl.Buffer(RGBA{Float32}, ctx, :w,            length(image))
kernel_buff  = cl.Buffer(Float32,       ctx, (:r, :copy),   hostbuf=kernel)

# build the program and construct a kernel object


# const global float4 *src_buf,
# const int src_width,
# global float4 *dst_buf,
# constant float *cmatrix,
# const int matrix_length,
# const int yoff

# call the kernel object with global size set to the size our arrays
@show gauss_cl_func
gauss_cl_func[queue, size(image)](  img_buff, Cint(size(image, 1)),
                                    blurred_buff,
                                    kernel_buff, Cint(length(kernel)),
                                    Cint(0));

# perform a blocking read of the result from the device
result_img = reshape(cl.read(queue, blurred_buff), size(image))

Base.clamp(x::Float32) = isnan(x) ? U8(0) : U8(clamp(x, 0.0f0, 1.0f0))

img = map(result_img) do x
    RGBA{U8}(clamp(red(x)), clamp(green(x)), clamp(blue(x)), clamp(alpha(x)))
end
# save("test.png", img)
using ImageView
imshow(img)
sleep(10)
