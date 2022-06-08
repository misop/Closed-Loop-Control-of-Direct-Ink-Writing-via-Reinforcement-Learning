#pragma once

#include "../../core/maths.h"

#include <glad/gl.h>
#include <GL/gl.h>
#include <GL/freeglut.h>

#include <vector>

#define STRINGIFY(A) #A
#define glVerify(x) x


// Ellipsoid shaders
//
#pragma region vertEllipsoidDepthShader
const char *vertEllipsoidDepthShader = "#version 460\n" STRINGIFY(

layout(location = 0) in vec4 Q;
layout(location = 1) in vec4 q1;
layout(location = 2) in vec4 q2;
layout(location = 3) in vec4 q3;

layout(location = 0) uniform mat4 V;
layout(location = 1) uniform mat4 P;

layout(location = 0) out VS_OUT {
    vec4 texCoord0;
    vec4 texCoord1;
    vec4 texCoord2;
    vec4 texCoord3;
    vec4 texCoord4;
    vec4 texCoord5;
} vs_out;

float Sign(float x) { return x < 0.0 ? -1.0: 1.0; }

bool solveQuadratic(float a, float b, float c, out float minT, out float maxT) {
	if (a == 0.0 && b == 0.0) {
		minT = maxT = 0.0;
		return false;
	}

	float discriminant = b*b - 4.0*a*c;

	if (discriminant < 0.0) {
		return false;
	}

	float t = -0.5*(b + Sign(b)*sqrt(discriminant));
	minT = t / a;
	maxT = c / t;

	if (minT > maxT) {
		float tmp = minT;
		minT = maxT;
		maxT = tmp;
	}

	return true;
}

float DotInvW(vec4 a, vec4 b) {	return a.x*b.x + a.y*b.y + a.z*b.z - a.w*b.w; }

void main() {
	gl_PointSize = 2;
	gl_Position = P*V*Q;
}
);
#pragma endregion

#pragma region fragEllipsoidDepthShader
const char *fragEllipsoidDepthShader = "#version 460\n" STRINGIFY(

layout(location = 0) in GS_OUT {
    vec4 texCoord0;
    vec4 texCoord1;
    vec4 texCoord2;
    vec4 texCoord3;
} fs_in;

layout(location = 2) uniform mat4 PM;
layout(location = 3) uniform vec3 invViewport;
layout(location = 4) uniform vec3 invProjection;

out vec4 fragColor;

float Sign(float x) { return x < 0.0 ? -1.0: 1.0; }

bool solveQuadratic(float a, float b, float c, out float minT, out float maxT) {
	if (a == 0.0 && b == 0.0) {
		minT = maxT = 0.0;
		return true;
	}

	float discriminant = b*b - 4.0*a*c;

	if (discriminant < 0.0) {
		return false;
	}

	float t = -0.5*(b + Sign(b)*sqrt(discriminant));
	minT = t / a;
	maxT = c / t;

	if (minT > maxT) {
		float tmp = minT;
		minT = maxT;
		maxT = tmp;
	}

	return true;
}

float sqr(float x) { return x*x; }

void main() {
	fragColor = vec4(1.0-gl_FragCoord.z);
	return;
}
);
#pragma endregion

// vertex shader
#pragma region vertexShader
const char *vertexShader = "#version 130\n" STRINGIFY(

uniform mat4 lightTransform; 
uniform vec3 lightDir;
uniform float bias;
uniform vec4 clipPlane;
uniform float expand;

uniform mat4 objectTransform;

void main()
{
	vec3 n = normalize((objectTransform*vec4(gl_Normal, 0.0)).xyz);
	vec3 p = (objectTransform*vec4(gl_Vertex.xyz, 1.0)).xyz;

    // calculate window-space point size
	gl_Position = gl_ModelViewProjectionMatrix * vec4(p + expand*n, 1.0);

	gl_TexCoord[0].xyz = n;
	gl_TexCoord[1] = lightTransform*vec4(p + n*bias, 1.0);
	gl_TexCoord[2] = gl_ModelViewMatrix*vec4(lightDir, 0.0);
	gl_TexCoord[3].xyz = p;
	gl_TexCoord[4] = gl_Color;
	gl_TexCoord[5] = gl_MultiTexCoord0;
	gl_TexCoord[6] = gl_SecondaryColor;
	gl_TexCoord[7] = gl_ModelViewMatrix*vec4(gl_Vertex.xyz, 1.0);

	gl_ClipDistance[0] = dot(clipPlane,vec4(gl_Vertex.xyz, 1.0));
}
);
#pragma endregion

// pixel shader for rendering points as shaded spheres
#pragma region fragmentShader
const char *fragmentShader = STRINGIFY(

uniform vec3 lightDir;
uniform vec3 lightPos;
uniform float spotMin;
uniform float spotMax;
uniform vec3 color;
uniform vec4 fogColor;

uniform sampler2DShadow shadowTex;
uniform vec2 shadowTaps[12];

uniform sampler2D tex;
uniform bool sky;

uniform bool grid;
uniform bool texture;

float sqr(float x) { return x*x; }

// sample shadow map
float shadowSample()
{
	vec3 pos = vec3(gl_TexCoord[1].xyz/gl_TexCoord[1].w);
	vec3 uvw = (pos.xyz*0.5)+vec3(0.5);

	// user clip
	if (uvw.x  < 0.0 || uvw.x > 1.0)
		return 1.0;
	if (uvw.y < 0.0 || uvw.y > 1.0)
		return 1.0;
	
	float s = 0.0;
	float radius = 0.002;

	const int numTaps = 12;

	for (int i=0; i < numTaps; i++)
	{
		s += shadow2D(shadowTex, vec3(uvw.xy + shadowTaps[i]*radius, uvw.z)).r;
	}

	s /= numTaps;
	return s;
}

float filterwidth(vec2 v)
{
  vec2 fw = max(abs(dFdx(v)), abs(dFdy(v)));
  return max(fw.x, fw.y);
}

vec2 bump(vec2 x) 
{
	return (floor((x)/2) + 2.f * max(((x)/2) - floor((x)/2) - .5f, 0.f)); 
}

float checker(vec2 uv)
{
  float width = filterwidth(uv);
  vec2 p0 = uv - 0.5 * width;
  vec2 p1 = uv + 0.5 * width;
  
  vec2 i = (bump(p1) - bump(p0)) / width;
  return i.x * i.y + (1 - i.x) * (1 - i.y);
}

void main()
{
    // calculate lighting
	float shadow = max(shadowSample(), 0.5);

	vec3 lVec = normalize(gl_TexCoord[3].xyz-(lightPos));
	vec3 lPos = vec3(gl_TexCoord[1].xyz/gl_TexCoord[1].w);
	float attenuation = max(smoothstep(spotMax, spotMin, dot(lPos.xy, lPos.xy)), 0.05);
		
	vec3 n = gl_TexCoord[0].xyz;
	vec3 color = gl_TexCoord[4].xyz;

	if (!gl_FrontFacing)
	{
		color = gl_TexCoord[6].xyz;
		n *= -1.0f;
	}

	if (grid && (n.y >0.995))
	{
		color *= 1.0 - 0.25 * checker(vec2(gl_TexCoord[3].x, gl_TexCoord[3].z));
	}
	else if (grid && abs(n.z) > 0.995)
	{
		color *= 1.0 - 0.25 * checker(vec2(gl_TexCoord[3].y, gl_TexCoord[3].x));
	}

	if (texture)
	{
		color = texture2D(tex, gl_TexCoord[5].xy).xyz;
	}
	
	// direct light term
	float wrap = 0.0;
	vec3 diffuse = color*vec3(1.0, 1.0, 1.0)*max(0.0, (-dot(lightDir, n)+wrap)/(1.0+wrap)*shadow)*attenuation;
	
	// wrap ambient term aligned with light dir
	vec3 light = vec3(0.03, 0.025, 0.025)*1.5;
	vec3 dark = vec3(0.025, 0.025, 0.03);
	vec3 ambient = 4.0*color*mix(dark, light, -dot(lightDir, n)*0.5 + 0.5)*attenuation;

	vec3 fog = mix(vec3(fogColor), diffuse + ambient, exp(gl_TexCoord[7].z*fogColor.w));

	gl_FragColor = vec4(pow(fog, vec3(1.0/2.2)), 1.0);				
}
);
#pragma endregion