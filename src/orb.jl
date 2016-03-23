import OpenCL
using Colors, FileIO


a = load(joinpath(homedir(), "surface.png"))
img_size = size(a)

const cl = OpenCL
device, ctx, queue = cl.create_compute_context()
const orb_kernel_src = load_kernel("orb.cl")

# create opencl buffer objects
# copies to the device initiated when the kernel function is called
img_buff = cl.Buffer(RGB{U8}, ctx, (:r, :copy), hostbuf=a)
responses = cl.Buffer(Float32, ctx, :w, length(a))
n_keypoints = 50
keypoints = cl.Buffer(Cint, ctx, :w, n_keypoints)

# build the program and construct a kernel object
orb_program = cl.build!(cl.Program(ctx, source=orb_kernel_src))
orb_kernel = cl.Kernel(orb_program, "orb")

uchar* imgbuf,
int imgstep,
int imgoffset0,
int* layerinfo,
int* keypoints,
float* responses,
int nkeypoints

# call the kernel object with global size set to the size our arrays
sum_kernel[queue, size(a)](a_buff, b_buff, c_buff)

# perform a blocking read of the result from the device
r = cl.read(queue, c_buff)

# check to see if our result is what we expect!
if isapprox(norm(r - (a+b)), zero(Float32))
    info("Success!")
else
    error("Norm should be 0.0f")
end
