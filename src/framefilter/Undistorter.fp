// Shader for radial undistortion only

in vec2 uv;     // 2D pixel coord
out vec4 color; // Color of pixel at uv

// Parameters
uniform sampler_2d
uniform texture_sampler;
uniform vec2 focal_length;
uniform vec2 optical_center;
uniform vec3 rad_coeffs;
//uniform vec3 tang_coeffs;
uniform vec2 frame_size;

void main 
{
    vec2 lens_coords = (uv - optical_center) / focal_length;

    float r2 = dot(lens_coords, lens_coords);
    float r4 = r2 * r2; 
    float r6 = r4 * r2;
    //float xy = lens_coords.x * lens_coords.y;

    lens_undist = lens_coords * (1 + rad_coeffs.x * r2 + rad_coeffs.y * r4 + rad_coeffs.z * r6);
    uv_undist = (focal_length * lens_undist + optical_center);

    color = textur2D(texture_sampler, uv_undist);
}
