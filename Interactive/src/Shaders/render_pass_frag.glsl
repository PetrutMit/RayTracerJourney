#version 330 core

in vec2 TexCoords;

layout (location = 0) out vec4 color;

uniform sampler2D accumulatedTexture;

void main()
{
	color = vec4(texture(accumulatedTexture, TexCoords).rgb, 1.0);
}
