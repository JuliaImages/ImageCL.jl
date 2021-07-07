using OpenCL
import  OpenCL: cl
using Colors, FileIO, Images
using ImageCL
using ImageCL.GaussianBlur
using Images
using ImageView

image = map(Colors.RGBA{Float32}, load(joinpath(dirname(@__FILE__()), "../assets/cups.jpg")))
imshow(image)

result_img = blur(  image;
                    kernelSigmas = (2.0, 2.0),
                    kernelSize = (5,5))

Base.clamp(x::Float32) = isnan(x) ? U8(0) : U8(clamp(x, 0.0f0, 1.0f0))

img = map(result_img) do x
    RGBA{U8}(clamp(red(x)), clamp(green(x)), clamp(blue(x)), clamp(alpha(x)))
end
# save("test.png", img)
using ImageView
imshow(img)

sleep(20)
