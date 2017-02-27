// taken from https://github.com/seung-lab/opencl-imageproc/blob/master/resize.py

#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__kernel void nearest2D(__read_only image2d_t src, __write_only image2d_t dst) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 dst_dim = get_image_dim(dst);
    if (pos.x >= dst_dim.x || pos.y >= dst_dim.y) {
        return;
    }
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const float2 rescale_ratio = convert_float2(get_image_dim(src)) / convert_float2(get_image_dim(dst));
    const float2 samplepos = rescale_ratio * (convert_float2(pos) + 0.5f);
    const uint4 pix = read_imageui(src, sampler, convert_int2_sat_rtz(samplepos));
    write_imageui(dst, pos, pix);
}
__kernel void linear2D(__read_only image2d_t src, __write_only image2d_t dst) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 dst_dim = get_image_dim(dst);
    if (pos.x >= dst_dim.x || pos.y >= dst_dim.y) {
        return;
    }
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const float2 rescale_ratio = convert_float2(get_image_dim(src)) / convert_float2(get_image_dim(dst));
    const float2 sample_pos = rescale_ratio * (convert_float2(pos) + 0.5f);
    const int2 read_pos = convert_int2_sat_rtz(sample_pos);
    const float2 sample_ratio = sample_pos - convert_float2(read_pos);
    // Interpolating along X
    const float4 pixY0 = mix(convert_float4_rtz(read_imageui(src, sampler, read_pos + (int2)(0,0))),
                             convert_float4_rtz(read_imageui(src, sampler, read_pos + (int2)(1,0))),
                             sample_ratio.x);
    const float4 pixY1 = mix(convert_float4_rtz(read_imageui(src, sampler, read_pos + (int2)(0,1))),
                             convert_float4_rtz(read_imageui(src, sampler, read_pos + (int2)(1,1))),
                             sample_ratio.x);
    // Interpolating along Y
    const uint4 pix = convert_uint4_sat_rtz(mix(pixY0, pixY1, sample_ratio.y) + 0.5f);
    write_imageui(dst, pos, pix);
}
__kernel void nearest3D(__read_only image3d_t src, __write_only image3d_t dst) {
    const int4 pos = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    const int4 dst_dim = get_image_dim(dst);
    if (pos.x >= dst_dim.x || pos.y >= dst_dim.y || pos.z >= dst_dim.z) {
        return;
    }
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const float4 rescale_ratio = convert_float4(get_image_dim(src)) / convert_float4(get_image_dim(dst));
    const float4 samplepos = rescale_ratio * (convert_float4(pos) + 0.5f);
    const uint4 pix = read_imageui(src, sampler, convert_int4_sat_rtz(samplepos));
    write_imageui(dst, pos, pix);
}
__kernel void linear3D(__read_only image3d_t src, __write_only image3d_t dst) {
    const int4 pos = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    const int4 dst_dim = get_image_dim(dst);
    if (pos.x >= dst_dim.x || pos.y >= dst_dim.y || pos.z >= dst_dim.z) {
        return;
    }
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const float4 rescale_ratio = convert_float4(get_image_dim(src)) / convert_float4(get_image_dim(dst));
    const float4 sample_pos = rescale_ratio * (convert_float4(pos) + 0.5f);
    const int4 read_pos = convert_int4_sat_rtz(sample_pos);
    const float4 sample_ratio = sample_pos - convert_float4(read_pos);
    // Interpolating along X
    const float4 pix000_100 = mix(convert_float4_rtz(read_imageui(src, sampler, read_pos + (int4)(0,0,0,0))),
                            convert_float4_rtz(read_imageui(src, sampler, read_pos + (int4)(1,0,0,0))),
                            sample_ratio.x);
    const float4 pix010_110 = mix(convert_float4_rtz(read_imageui(src, sampler, read_pos + (int4)(0,1,0,0))),
                            convert_float4_rtz(read_imageui(src, sampler, read_pos + (int4)(1,1,0,0))),
                            sample_ratio.x);
    const float4 pix001_101 = mix(convert_float4_rtz(read_imageui(src, sampler, read_pos + (int4)(0,0,1,0))),
                            convert_float4_rtz(read_imageui(src, sampler, read_pos + (int4)(1,0,1,0))),
                            sample_ratio.x);
    const float4 pix011_111 = mix(convert_float4_rtz(read_imageui(src, sampler, read_pos + (int4)(0,1,1,0))),
                            convert_float4_rtz(read_imageui(src, sampler, read_pos + (int4)(1,1,1,0))),
                            sample_ratio.x);
    // Interpolating along Y
    const float4 pixZ0 = mix(pix000_100, pix010_110, sample_ratio.y);
    const float4 pixZ1 = mix(pix001_101, pix011_111, sample_ratio.y);
    // Interpolating along Z
    const uint4 pix = convert_uint4_sat_rtz(mix(pixZ0, pixZ1, sample_ratio.z) + 0.5f);
    write_imageui(dst, pos, pix);
}
