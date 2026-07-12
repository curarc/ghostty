#include "common.glsl"

layout(binding = 0) uniform sampler2DRect atlas_grayscale;
layout(binding = 1) uniform sampler2DRect atlas_color;

// Per-cell background colors are looked up in the fragment shader
// rather than the vertex shader for driver compatibility — some
// drivers set GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS to 0 even on
// OpenGL 4.3+, while fragment shader SSBO support is universal.
layout(binding = 1, std430) readonly buffer bg_cells {
    uint bg_colors[];
};

in CellTextVertexOut {
    flat uint atlas;
    flat vec4 color;
    flat vec4 bg_color;
    flat uvec2 grid_pos;
    flat uint glyph_bools;
    vec2 tex_coord;
} in_data;

// Values `atlas` can take.
const uint ATLAS_GRAYSCALE = 0u;
const uint ATLAS_COLOR = 1u;

// Masks for `glyph_bools`, matching the vertex shader.
const uint NO_MIN_CONTRAST = 1u;
const uint IS_CURSOR_GLYPH = 2u;

// Must declare this output for some versions of OpenGL.
layout(location = 0) out vec4 out_FragColor;

void main() {
    bool use_linear_blending = (bools & USE_LINEAR_BLENDING) != 0;
    bool use_linear_correction = (bools & USE_LINEAR_CORRECTION) != 0;

    switch (in_data.atlas) {
        default:
        case ATLAS_GRAYSCALE:
        {
            // Our input color is always linear.
            vec4 color = in_data.color;

            // If we're not doing linear blending, then we need to
            // re-apply the gamma encoding to our color manually.
            //
            // Since the alpha is premultiplied, we need to divide
            // it out before unlinearizing and re-multiply it after.
            if (!use_linear_blending) {
                color.rgb /= vec3(color.a);
                color = unlinearize(color);
                color.rgb *= vec3(color.a);
            }

            // Compute per-cell background color from the storage buffer.
            // We do this in the fragment shader because some drivers (notably
            // llvmpipe / software rasterizers) set
            // GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS = 0, making the SSBO
            // inaccessible from vertex shaders. Fragment shader SSBO support
            // is universally available wherever GL 4.3 is supported.
            uvec2 gs = unpack2u16(grid_size_packed_2u16);
            vec4 cell_bg_color = load_color(
                    unpack4u8(bg_colors[in_data.grid_pos.y * gs.x + in_data.grid_pos.x]),
                    true
                );
            vec4 global_bg_vec = load_color(
                    unpack4u8(bg_color_packed_4u8),
                    true
                );
            vec4 bg = cell_bg_color + global_bg_vec * vec4(1.0 - cell_bg_color.a);

            // Apply minimum contrast against the computed cell background.
            if (min_contrast > 1.0f && (in_data.glyph_bools & NO_MIN_CONTRAST) == 0u) {
                color = contrasted_color(min_contrast, color, bg);
            }

            // Apply cursor color override after minimum contrast.
            uvec2 cursor_pos = unpack2u16(cursor_pos_packed_2u16);
            bool cursor_wide = (bools & CURSOR_WIDE) != 0;
            bool is_cursor_pos = ((in_data.grid_pos.x == cursor_pos.x) ||
                    (cursor_wide && (in_data.grid_pos.x == (cursor_pos.x + 1u)))) &&
                (in_data.grid_pos.y == cursor_pos.y);
            if ((in_data.glyph_bools & IS_CURSOR_GLYPH) == 0u && is_cursor_pos) {
                color = load_color(unpack4u8(cursor_color_packed_4u8), use_linear_blending);
            }

            // Fetch our alpha mask for this pixel.
            float a = texture(atlas_grayscale, in_data.tex_coord).r;

            // Linear blending weight correction corrects the alpha value to
            // produce blending results which match gamma-incorrect blending.
            if (use_linear_correction) {
                // Short explanation of how this works:
                //
                // We get the luminances of the foreground and background colors,
                // and then unlinearize them and perform blending on them. This
                // gives us our desired luminance, which we derive our new alpha
                // value from by mapping the range [bg_l, fg_l] to [0, 1], since
                // our final blend will be a linear interpolation from bg to fg.
                //
                // This yields virtually identical results for grayscale blending,
                // and very similar but non-identical results for color blending.
                float fg_l = luminance(color.rgb);
                float bg_l = luminance(bg.rgb);
                // To avoid numbers going haywire, we don't apply correction
                // when the bg and fg luminances are within 0.001 of each other.
                if (abs(fg_l - bg_l) > 0.001) {
                    float blend_l = linearize(unlinearize(fg_l) * a + unlinearize(bg_l) * (1.0 - a));
                    a = clamp((blend_l - bg_l) / (fg_l - bg_l), 0.0, 1.0);
                }
            }

            // Multiply our whole color by the alpha mask.
            // Since we use premultiplied alpha, this is
            // the correct way to apply the mask.
            color *= a;

            out_FragColor = color;
            return;
        }

        case ATLAS_COLOR:
        {
            // For now, we assume that color glyphs
            // are already premultiplied linear colors.
            vec4 color = texture(atlas_color, in_data.tex_coord);

            // If we are doing linear blending, we can return this right away.
            if (use_linear_blending) {
                out_FragColor = color;
                return;
            }

            // Otherwise we need to unlinearize the color. Since the alpha is
            // premultiplied, we need to divide it out before unlinearizing.
            color.rgb /= vec3(color.a);
            color = unlinearize(color);
            color.rgb *= vec3(color.a);

            out_FragColor = color;
            return;
        }
    }
}
