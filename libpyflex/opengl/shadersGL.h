#pragma once

#include <NvFlex.h>
#include <NvFlexExt.h>
#include <NvFlexDevice.h>

#include "../../core/mesh.h"
#include "../../core/tga.h"	
#include "../../core/platform.h"
#include "../../core/extrude.h"

#include "./opengl/shader.h"
#include "SimBuffers.h"

namespace OpenGL {
	typedef unsigned int VertexBuffer;
	typedef GLuint VertexArrayObject;
	typedef unsigned int IndexBuffer;
	typedef unsigned int Texture;

	struct FluidRenderer {
		GLuint mDepthFbo;
		GLuint mDepthTex;
		GLuint mDepthSmoothTex;
		GLuint mSceneFbo;
		GLuint mSceneTex;
		GLuint mReflectTex;

		GLuint mThicknessFbo;
		GLuint mThicknessTex;

		GLuint mPointThicknessProgram;

		GLuint mEllipsoidThicknessProgram;
		GLuint mEllipsoidDepthProgram;

		GLuint mCompositeProgram;
		GLuint mDepthBlurProgram;

		int mSceneWidth;
		int mSceneHeight;
	};

	struct FluidRenderBuffers {
		FluidRenderBuffers(int numParticles = 0):
			mVAO(0),
			mPositionVBO(0),
			mDensityVBO(0),
			mIndices(0),
			mPositionBuf(nullptr),
			mDensitiesBuf(nullptr),
			mIndicesBuf(nullptr) {
			mNumParticles = numParticles;
			for (int i = 0; i < 3; i++) { 
				mAnisotropyVBO[i] = 0;
				mAnisotropyBuf[i] = nullptr;
			}
		}
		~FluidRenderBuffers() {
			glDeleteVertexArrays(1, &mVAO);

			glDeleteBuffers(1, &mPositionVBO);
			glDeleteBuffers(3, mAnisotropyVBO);
			glDeleteBuffers(1, &mDensityVBO);
			glDeleteBuffers(1, &mIndices);

			NvFlexUnregisterOGLBuffer(mPositionBuf);
			NvFlexUnregisterOGLBuffer(mDensitiesBuf);
			NvFlexUnregisterOGLBuffer(mIndicesBuf);

			NvFlexUnregisterOGLBuffer(mAnisotropyBuf[0]);
			NvFlexUnregisterOGLBuffer(mAnisotropyBuf[1]);
			NvFlexUnregisterOGLBuffer(mAnisotropyBuf[2]);
		}

		int mNumParticles;
		VertexBuffer mPositionVBO;
		VertexBuffer mDensityVBO;
		VertexBuffer mAnisotropyVBO[3];
		IndexBuffer mIndices;
		VertexArrayObject mVAO;

		// wrapper buffers that allow Flex to write directly to VBOs
		NvFlexBuffer* mPositionBuf;
		NvFlexBuffer* mDensitiesBuf;
		NvFlexBuffer* mAnisotropyBuf[3];
		NvFlexBuffer* mIndicesBuf;
	};

	struct DiffuseRenderBuffers {
		DiffuseRenderBuffers(int numParticles = 0):
			mDiffusePositionVBO(0),
			mDiffuseVelocityVBO(0),
			mDiffuseIndicesIBO(0),
			mDiffuseIndicesBuf(nullptr),
			mDiffusePositionsBuf(nullptr),
			mDiffuseVelocitiesBuf(nullptr)
		{
			mNumParticles = numParticles;
		}
		~DiffuseRenderBuffers() {
			if (mNumParticles > 0) {
				glDeleteBuffers(1, &mDiffusePositionVBO);
				glDeleteBuffers(1, &mDiffuseVelocityVBO);
				glDeleteBuffers(1, &mDiffuseIndicesIBO);

				NvFlexUnregisterOGLBuffer(mDiffuseIndicesBuf);
				NvFlexUnregisterOGLBuffer(mDiffusePositionsBuf);
				NvFlexUnregisterOGLBuffer(mDiffuseVelocitiesBuf);
			}
		}

		int mNumParticles;
		VertexBuffer mDiffusePositionVBO;
		VertexBuffer mDiffuseVelocityVBO;
		IndexBuffer mDiffuseIndicesIBO;

		NvFlexBuffer* mDiffuseIndicesBuf;
		NvFlexBuffer* mDiffusePositionsBuf;
		NvFlexBuffer* mDiffuseVelocitiesBuf;
	};

	struct MSAABuffers {
		GLuint g_msaaFbo;
		GLuint g_msaaColorBuf;
		GLuint g_msaaDepthBuf;

		MSAABuffers() : g_msaaFbo(0), g_msaaColorBuf(0), g_msaaDepthBuf(0) {}
	};

	struct ShadowMap {
		GLuint texture;
		GLuint framebuffer;
	};

	struct GpuMesh {
		GLuint mPositionsVBO;
		GLuint mNormalsVBO;
		GLuint mIndicesIBO;

		int mNumVertices;
		int mNumFaces;
	};

	void GlslPrintShaderLog(GLuint obj) {
		int infologLength = 0;
		int charsWritten = 0;
		char *infoLog;

		GLint result;
		glGetShaderiv(obj, GL_COMPILE_STATUS, &result);

		// only print log if compile fails
		if (result == GL_FALSE) {
			glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

			if (infologLength > 1) {
				infoLog = (char *)malloc(infologLength);
				glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
				printf("%s\n", infoLog);
				free(infoLog);
			}
		}
	}

	GLuint CompileProgram(const char *vsource, const char *fsource, const char* gsource = NULL) {
		GLuint vertexShader = GLuint(-1);
		GLuint geometryShader = GLuint(-1);
		GLuint fragmentShader = GLuint(-1);

		GLuint program = glCreateProgram();

		if (vsource) {
			vertexShader = glCreateShader(GL_VERTEX_SHADER);
			glShaderSource(vertexShader, 1, &vsource, 0);
			glCompileShader(vertexShader);
			GlslPrintShaderLog(vertexShader);
			glAttachShader(program, vertexShader);
		}

		if (fsource) {
			fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
			glShaderSource(fragmentShader, 1, &fsource, 0);
			glCompileShader(fragmentShader);
			GlslPrintShaderLog(fragmentShader);
			glAttachShader(program, fragmentShader);
		}

		if (gsource) {
			geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
			glShaderSource(geometryShader, 1, &gsource, 0);
			glCompileShader(geometryShader);
			GlslPrintShaderLog(geometryShader);

			// hack, force billboard gs mode
			glAttachShader(program, geometryShader);
			glProgramParameteriEXT(program, GL_GEOMETRY_VERTICES_OUT_EXT, 4);
			glProgramParameteriEXT(program, GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
			glProgramParameteriEXT(program, GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
		}

		glLinkProgram(program);

		// check if program linked
		GLint success = 0;
		glGetProgramiv(program, GL_LINK_STATUS, &success);

		if (!success) {
			char temp[256];
			glGetProgramInfoLog(program, 256, 0, temp);
			printf("Failed to link program:\n%s\n", temp);
			glDeleteProgram(program);
			program = 0;
		}

		if (vsource) glDeleteShader(vertexShader);
		if (fsource) glDeleteShader(fragmentShader);
		if (gsource) glDeleteShader(geometryShader);

		return program;
	}

	void ReshapeRender(MSAABuffers &g_msaa, int width, int height) {
		int g_msaaSamples = 8;
		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));

		if (g_msaa.g_msaaFbo) {
			glVerify(glDeleteFramebuffers(1, &g_msaa.g_msaaFbo));
			glVerify(glDeleteRenderbuffers(1, &g_msaa.g_msaaColorBuf));
			glVerify(glDeleteRenderbuffers(1, &g_msaa.g_msaaDepthBuf));
		}

		int samples;
		glGetIntegerv(GL_MAX_SAMPLES_EXT, &samples);

		// clamp samples to 4 to avoid problems with point sprite scaling
		samples = Min(samples, Min(g_msaaSamples, 4));

		glVerify(glGenFramebuffers(1, &g_msaa.g_msaaFbo));
		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, g_msaa.g_msaaFbo));

		glVerify(glGenRenderbuffers(1, &g_msaa.g_msaaColorBuf));
		glVerify(glBindRenderbuffer(GL_RENDERBUFFER, g_msaa.g_msaaColorBuf));
		glVerify(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, width, height));

		glVerify(glGenRenderbuffers(1, &g_msaa.g_msaaDepthBuf));
		glVerify(glBindRenderbuffer(GL_RENDERBUFFER, g_msaa.g_msaaDepthBuf));
		glVerify(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT, width, height));
		glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g_msaa.g_msaaDepthBuf));

		glVerify(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, g_msaa.g_msaaColorBuf));

		glVerify(glCheckFramebufferStatus(GL_FRAMEBUFFER));

		glEnable(GL_MULTISAMPLE);
	}

	FluidRenderer* CreateFluidRenderer(GLuint g_msaaFbo, uint32_t width, uint32_t height, bool thickness) {
		FluidRenderer* renderer = new FluidRenderer();

		renderer->mSceneWidth = width;
		renderer->mSceneHeight = height;

		renderer->mEllipsoidDepthProgram = CompileProgram(vertEllipsoidDepthShader, fragEllipsoidDepthShader);

		return renderer;
	}

	FluidRenderBuffers* CreateFluidRenderBuffers(NvFlexLibrary* g_flexLib, int numFluidParticles, bool enableInterop) {
		FluidRenderBuffers* buffers = new FluidRenderBuffers(numFluidParticles);
		
		// vbos
		glVerify(glGenBuffers(1, &buffers->mPositionVBO));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mPositionVBO));
		glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * numFluidParticles, 0, GL_DYNAMIC_DRAW));

		// density
		glVerify(glGenBuffers(1, &buffers->mDensityVBO));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mDensityVBO));
		glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(int)*numFluidParticles, 0, GL_DYNAMIC_DRAW));

		for (int i = 0; i < 3; ++i) {
			glVerify(glGenBuffers(1, &buffers->mAnisotropyVBO[i]));
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mAnisotropyVBO[i]));
			glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * numFluidParticles, 0, GL_DYNAMIC_DRAW));
		}

		glVerify(glGenBuffers(1, &buffers->mIndices));
		glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers->mIndices));
		glVerify(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*numFluidParticles, 0, GL_DYNAMIC_DRAW));

		if (enableInterop) {
			buffers->mPositionBuf = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mPositionVBO, numFluidParticles, sizeof(Vec4));
			buffers->mDensitiesBuf = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mDensityVBO, numFluidParticles, sizeof(float));
			buffers->mIndicesBuf = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mIndices, numFluidParticles, sizeof(int));

			buffers->mAnisotropyBuf[0] = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mAnisotropyVBO[0], numFluidParticles, sizeof(Vec4));
			buffers->mAnisotropyBuf[1] = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mAnisotropyVBO[1], numFluidParticles, sizeof(Vec4));
			buffers->mAnisotropyBuf[2] = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mAnisotropyVBO[2], numFluidParticles, sizeof(Vec4));
		}

		glGenVertexArrays(1, &buffers->mVAO);
		glBindVertexArray(buffers->mVAO);
		glBindBuffer(GL_ARRAY_BUFFER, buffers->mPositionVBO);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		for (int i = 0; i < 3; i++) {
			glBindBuffer(GL_ARRAY_BUFFER, buffers->mAnisotropyVBO[i]);
			glVertexAttribPointer(1+i, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		}
		for (int i = 0; i < 4; i++) {
			glEnableVertexAttribArray(i);
		}

		return buffers;
	}

	void DestroyFluidRenderBuffers(FluidRenderBuffers* buffers) {
		delete buffers;
	}

	ShadowMap* ShadowCreate() {
		const int kShadowResolution = 2048;

		GLuint texture;
		GLuint framebuffer;

		glVerify(glGenFramebuffers(1, &framebuffer));
		glVerify(glGenTextures(1, &texture));
		glVerify(glBindTexture(GL_TEXTURE_2D, texture));

		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)); 
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); 
		
		// This is to allow usage of shadow2DProj function in the shader 
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)); 
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)); 
		glVerify(glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY)); 

		glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, kShadowResolution, kShadowResolution, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL));

		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, framebuffer));

		glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture, 0));

		ShadowMap* map = new ShadowMap();
		map->texture = texture;
		map->framebuffer = framebuffer;

		return map;
	}

	DiffuseRenderBuffers* CreateDiffuseRenderBuffers(NvFlexLibrary* g_flexLib, int numDiffuseParticles, bool& enableInterop) {
		DiffuseRenderBuffers* buffers = new DiffuseRenderBuffers(numDiffuseParticles);
		
		if (numDiffuseParticles > 0) {
			glVerify(glGenBuffers(1, &buffers->mDiffusePositionVBO));
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mDiffusePositionVBO));
			glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * numDiffuseParticles, 0, GL_DYNAMIC_DRAW));
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));

			glVerify(glGenBuffers(1, &buffers->mDiffuseVelocityVBO));
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mDiffuseVelocityVBO));
			glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * numDiffuseParticles, 0, GL_DYNAMIC_DRAW));
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));

			if (enableInterop) {
				buffers->mDiffusePositionsBuf = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mDiffusePositionVBO, numDiffuseParticles, sizeof(Vec4));
				buffers->mDiffuseVelocitiesBuf = NvFlexRegisterOGLBuffer(g_flexLib, buffers->mDiffuseVelocityVBO, numDiffuseParticles, sizeof(Vec4));
			}
		}

		return buffers;
	}

	void DestroyDiffuseRenderBuffers(DiffuseRenderBuffers* buffers) {
		delete buffers;
	}

	GpuMesh* CreateGpuMesh(const Mesh* m) {
		GpuMesh* mesh = new GpuMesh();

		mesh->mNumVertices = m->GetNumVertices();
		mesh->mNumFaces = m->GetNumFaces();

		// vbos
		glVerify(glGenBuffers(1, &mesh->mPositionsVBO));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, mesh->mPositionsVBO));
		glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*m->m_positions.size(), &m->m_positions[0], GL_STATIC_DRAW));

		glVerify(glGenBuffers(1, &mesh->mNormalsVBO));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, mesh->mNormalsVBO));
		glVerify(glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*m->m_normals.size(), &m->m_normals[0], GL_STATIC_DRAW));

		glVerify(glGenBuffers(1, &mesh->mIndicesIBO));
		glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->mIndicesIBO));
		glVerify(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*m->m_indices.size(), &m->m_indices[0], GL_STATIC_DRAW));
		glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));

		return mesh;
	}

	void DestroyGpuMesh(GpuMesh* m) {
		glVerify(glDeleteBuffers(1, &m->mPositionsVBO));
		glVerify(glDeleteBuffers(1, &m->mNormalsVBO));
		glVerify(glDeleteBuffers(1, &m->mIndicesIBO));
	}

	void RenderFullscreenQuad() {
		glColor3f(1.0f, 1.0f, 1.0f);
		glBegin(GL_QUADS);

		glTexCoord2f(0.0f, 0.0f);
		glVertex2f(-1.0f, -1.0f);

		glTexCoord2f(1.0f, 0.0f);
		glVertex2f(1.0f, -1.0f);

		glTexCoord2f(1.0f, 1.0f);
		glVertex2f(1.0f, 1.0f);

		glTexCoord2f(0.0f, 1.0f);
		glVertex2f(-1.0f, 1.0f);

		glEnd();
	}

	void StartFrame(GLuint g_msaaFbo, Vec4 clearColor) {
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glDisable(GL_LIGHTING);
		glDisable(GL_BLEND);

		glPointSize(5.0f);

		glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
		glVerify(glClearColor(clearColor.x, clearColor.y, clearColor.z, 0.0f));
		glVerify(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
	}

	void EndFrame(GLuint g_msaaFbo, int g_screenWidth, int g_screenHeight) {
	}

	void ShadowApply(GLint sprogram, Vec3 lightPos, Vec3 lightTarget, Matrix44 lightTransform, GLuint shadowTex) {
		float g_shadowBias = 0.05f;
		
		GLint uLightTransform = glGetUniformLocation(sprogram, "lightTransform");
		glUniformMatrix4fv(uLightTransform, 1, false, lightTransform);

		GLint uLightPos = glGetUniformLocation(sprogram, "lightPos");
		glUniform3fv(uLightPos, 1, lightPos);
		
		GLint uLightDir = glGetUniformLocation(sprogram, "lightDir");
		glUniform3fv(uLightDir, 1, Normalize(lightTarget-lightPos));

		GLint uBias = glGetUniformLocation(sprogram, "bias");
		glUniform1f(uBias, g_shadowBias);

		const Vec2 taps[] = { 
			Vec2(-0.326212f,-0.40581f),Vec2(-0.840144f,-0.07358f),
			Vec2(-0.695914f,0.457137f),Vec2(-0.203345f,0.620716f),
			Vec2(0.96234f,-0.194983f),Vec2(0.473434f,-0.480026f),
			Vec2(0.519456f,0.767022f),Vec2(0.185461f,-0.893124f),
			Vec2(0.507431f,0.064425f),Vec2(0.89642f,0.412458f),
			Vec2(-0.32194f,-0.932615f),Vec2(-0.791559f,-0.59771f) 
		};
		
		GLint uShadowTaps = glGetUniformLocation(sprogram, "shadowTaps");
		glUniform2fv(uShadowTaps, 12, &taps[0].x);
		
		glEnable(GL_TEXTURE_2D);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, shadowTex);
	}

	void BindSolidShader(GLuint s_diffuseProgram, int g_screenWidth, int g_screenHeight, float g_spotMin, float g_spotMax, Vec3 lightPos, Vec3 lightTarget, Matrix44 lightTransform, ShadowMap* shadowMap, float bias, Vec4 fogColor) {
		glVerify(glViewport(0, 0, g_screenWidth, g_screenHeight));

		if (s_diffuseProgram == GLuint(-1))
			s_diffuseProgram = CompileProgram(vertexShader, fragmentShader);

		if (s_diffuseProgram) {
			glDepthMask(GL_TRUE);
			glEnable(GL_DEPTH_TEST);		

			glVerify(glUseProgram(s_diffuseProgram));
			glVerify(glUniform1i(glGetUniformLocation(s_diffuseProgram, "grid"), 0));
			glVerify(glUniform1f( glGetUniformLocation(s_diffuseProgram, "spotMin"), g_spotMin));
			glVerify(glUniform1f( glGetUniformLocation(s_diffuseProgram, "spotMax"), g_spotMax));
			glVerify(glUniform4fv( glGetUniformLocation(s_diffuseProgram, "fogColor"), 1, fogColor));

			glVerify(glUniformMatrix4fv(glGetUniformLocation(s_diffuseProgram, "objectTransform"), 1, false, Matrix44::kIdentity));

			// set shadow parameters
			ShadowApply(s_diffuseProgram, lightPos, lightTarget, lightTransform, shadowMap->texture);
		}
	}

	void UnbindSolidShader() {
		glActiveTexture(GL_TEXTURE1);
		glDisable(GL_TEXTURE_2D);
		glActiveTexture(GL_TEXTURE0);

		glUseProgram(0);
	}

	void SetView(Matrix44 view, Matrix44 proj) {
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(proj);

		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(view);
	}

	void SetCullMode(bool enabled) {
		if (enabled)
			glEnable(GL_CULL_FACE);		
		else
			glDisable(GL_CULL_FACE);		
	}

	void UpdateFluidRenderBuffers(FluidRenderBuffers* buffers, NvFlexSolver* solver, bool anisotropy, bool density) {
		// use VBO buffer wrappers to allow Flex to write directly to the OpenGL buffers
		// Flex will take care of any CUDA interop mapping/unmapping during the get() operations
		if (!anisotropy) {
			// regular particles
			NvFlexGetParticles(solver, buffers->mPositionBuf, NULL);
		} else {
			// fluid buffers
			NvFlexGetSmoothParticles(solver, buffers->mPositionBuf, NULL);
			NvFlexGetAnisotropy(solver, buffers->mAnisotropyBuf[0], buffers->mAnisotropyBuf[1], buffers->mAnisotropyBuf[2], NULL);
		}

		if (density) {
			NvFlexGetDensities(solver, buffers->mDensitiesBuf, NULL);
		} else {
			NvFlexGetPhases(solver, buffers->mDensitiesBuf, NULL);
		}

		NvFlexGetActive(solver, buffers->mIndicesBuf, NULL);
	}

	void UpdateFluidRenderBuffers(FluidRenderBuffers* buffers, Vec4* particles, float* densities, Vec4* anisotropy1, Vec4* anisotropy2, Vec4* anisotropy3, int numParticles, int* indices, int numIndices) {
		// regular particles
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mPositionVBO));
		glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, buffers->mNumParticles*sizeof(Vec4), particles));

		Vec4*const anisotropies[] = {
			anisotropy1,
			anisotropy2, 
			anisotropy3,
		};

		for (int i = 0; i < 3; i++) {
			Vec4* anisotropy = anisotropies[i];
			if (anisotropy) {
				glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mAnisotropyVBO[i]));
				glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, buffers->mNumParticles * sizeof(Vec4), anisotropy));
			}
		}

		// density /phase buffer
		if (densities) {
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, buffers->mDensityVBO));
			glVerify(glBufferSubData(GL_ARRAY_BUFFER, 0, buffers->mNumParticles*sizeof(float), densities));
		}

		if (indices) {
			glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers->mIndices));
			glVerify(glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, numIndices*sizeof(int), indices));
		}

		// reset
		glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
		glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
	}

	void DrawMesh(const Mesh* m, Vec3 color) {
		if (m) {
			glVerify(glColor3fv(color));
			glVerify(glSecondaryColor3fv(color));

			glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));
			glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));

			glVerify(glEnableClientState(GL_NORMAL_ARRAY));
			glVerify(glEnableClientState(GL_VERTEX_ARRAY));

			glVerify(glNormalPointer(GL_FLOAT, sizeof(float) * 3, &m->m_normals[0]));
			glVerify(glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, &m->m_positions[0]));

			if (m->m_colours.size()) {
				glVerify(glEnableClientState(GL_COLOR_ARRAY));
				glVerify(glColorPointer(4, GL_FLOAT, 0, &m->m_colours[0]));
			}

			glVerify(glDrawElements(GL_TRIANGLES, m->GetNumFaces() * 3, GL_UNSIGNED_INT, &m->m_indices[0]));

			glVerify(glDisableClientState(GL_VERTEX_ARRAY));
			glVerify(glDisableClientState(GL_NORMAL_ARRAY));

			if (m->m_colours.size())
				glVerify(glDisableClientState(GL_COLOR_ARRAY));
		}
	}

	void DrawGpuMesh(GpuMesh* m, const Matrix44& xform, const Vec3& color) {
		if (m) {
			GLint program;
			glGetIntegerv(GL_CURRENT_PROGRAM, &program);

			if (program)
				glUniformMatrix4fv( glGetUniformLocation(program, "objectTransform"), 1, false, xform);

			glVerify(glColor3fv(color));
			glVerify(glSecondaryColor3fv(color));

			glVerify(glEnableClientState(GL_VERTEX_ARRAY));
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, m->mPositionsVBO));
			glVerify(glVertexPointer(3, GL_FLOAT, sizeof(float)*3, 0));	

			glVerify(glEnableClientState(GL_NORMAL_ARRAY));
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, m->mNormalsVBO));
			glVerify(glNormalPointer(GL_FLOAT, sizeof(float)*3, 0));
			
			glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m->mIndicesIBO));

			glVerify(glDrawElements(GL_TRIANGLES, m->mNumFaces*3, GL_UNSIGNED_INT, 0));

			glVerify(glDisableClientState(GL_VERTEX_ARRAY));
			glVerify(glDisableClientState(GL_NORMAL_ARRAY));

			glVerify(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
			glVerify(glBindBuffer(GL_ARRAY_BUFFER, 0));	

			if (program)
				glUniformMatrix4fv(glGetUniformLocation(program, "objectTransform"), 1, false, Matrix44::kIdentity);
		}
	}

	void SetFillMode(bool wireframe) {
		glPolygonMode(GL_FRONT_AND_BACK, wireframe?GL_LINE:GL_FILL);
	}

	void DrawShapes(SimBuffers* g_buffers, std::map<NvFlexTriangleMeshId, OpenGL::GpuMesh*> &g_meshes, std::map<NvFlexConvexMeshId, OpenGL::GpuMesh*> &g_convexes) {
		for (int i = 0; i < g_buffers->shapeFlags.size(); ++i) {
			const int flags = g_buffers->shapeFlags[i];

			// unpack flags
			auto type = int(flags & eNvFlexShapeFlagTypeMask);

			Vec3 color = Vec3(0.9f);

			if (flags & eNvFlexShapeFlagTrigger) {
				color = Vec3(0.6f, 1.0, 0.6f);
				SetFillMode(true);
			}

			// render with prev positions to match particle update order
			// can also think of this as current/next
			const Quat rotation = g_buffers->shapePrevRotations[i];
			const Vec3 position = Vec3(g_buffers->shapePrevPositions[i]);

			NvFlexCollisionGeometry geo = g_buffers->shapeGeometry[i];

			if (type == eNvFlexShapeSphere) {
				Mesh* sphere = CreateSphere(20, 20, geo.sphere.radius);

				Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation));
				sphere->Transform(xform);

				DrawMesh(sphere, Vec3(color));

				delete sphere;
			} else if (type == eNvFlexShapeCapsule) {
				Mesh* capsule = CreateCapsule(10, 20, geo.capsule.radius, geo.capsule.halfHeight);

				// transform to world space
				Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*RotationMatrix(DegToRad(-90.0f), Vec3(0.0f, 0.0f, 1.0f));
				capsule->Transform(xform);

				DrawMesh(capsule, Vec3(color));

				delete capsule;
			} else if (type == eNvFlexShapeBox) {
				Mesh* box = CreateCubeMesh();

				Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(Vec3(geo.box.halfExtents)*2.0f);
				box->Transform(xform);

				DrawMesh(box, Vec3(color));
				delete box;
			} else if (type == eNvFlexShapeConvexMesh) {
				if (g_convexes.find(geo.convexMesh.mesh) != g_convexes.end()) {
					GpuMesh* m = g_convexes[geo.convexMesh.mesh];

					if (m) {
						Matrix44 xform = TranslationMatrix(Point3(g_buffers->shapePositions[i]))*RotationMatrix(Quat(g_buffers->shapeRotations[i]))*ScaleMatrix(geo.convexMesh.scale);
						DrawGpuMesh(m, xform, Vec3(color));
					}
				}
			} else if (type == eNvFlexShapeTriangleMesh) {
				if (g_meshes.find(geo.triMesh.mesh) != g_meshes.end()) {
					GpuMesh* m = g_meshes[geo.triMesh.mesh];

					if (m) {
						Matrix44 xform = TranslationMatrix(Point3(position))*RotationMatrix(Quat(rotation))*ScaleMatrix(geo.triMesh.scale);
						DrawGpuMesh(m, xform, Vec3(color));
					}
				}
			}
		}

		SetFillMode(false);
	}

	void RenderEllipsoids(SimBuffers* g_buffers, std::map<NvFlexTriangleMeshId, OpenGL::GpuMesh*> &g_meshes, std::map<NvFlexConvexMeshId, OpenGL::GpuMesh*> &g_convexes, GLuint g_msaaFbo, Mesh* g_mesh, FluidRenderer* render, FluidRenderBuffers* buffers, FluidRenderer* thickRender, Vec4 view_bounds, float g_spotMin, float g_spotMax, int n, int offset, float radius, float screenWidth, float screenAspect, float fov, Vec3 lightPos, Vec3 lightTarget, Matrix44 lightTransform, ShadowMap* shadowMap, Vec4 color, float blur, float ior, bool debug) {
		Matrix44 proj = OrthographicMatrix(view_bounds.x, view_bounds.y, view_bounds.z, view_bounds.w, 4.5f, 5.6f);
		Matrix44 view = RotationMatrix(DegToRad(90), Vec3(1.f, 0.f, 0.f))*TranslationMatrix(-Point3(0.f, 5.5f, 0.f));
		// fluid depth
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_DEPTH_TEST);
		
		glViewport(0, 0, int(screenWidth), int(screenWidth / screenAspect));
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(render->mEllipsoidDepthProgram);

		glUniformMatrix4fv(0, 1, false, view);
		glUniformMatrix4fv(1, 1, false, proj);
		
		float viewHeight = tanf(fov / 2.0f);
		glUniformMatrix4fv(2, 1, false, proj);
		glUniform3fv(3, 1, Vec3(1.0f / screenWidth, screenAspect / screenWidth, 1.0f));
		glUniform3fv(4, 1, Vec3(screenAspect*viewHeight, viewHeight, 1.0f));

		glBindVertexArray(buffers->mVAO);
		glDrawArrays(GL_POINTS, offset, n);
	}

}