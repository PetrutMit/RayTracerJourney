#version 330 core

in vec2 TexCoords;

layout (location = 0) out vec4 color;

uniform sampler2D screenTexture;

void main()
{
    color = vec4(texture(screenTexture, TexCoords).rgb, 1.0);
}
