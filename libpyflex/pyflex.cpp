#include <glad/egl.h>
#include <glad/gl.h>

#include <NvFlex.h>
#include <NvFlexExt.h>
#include <NvFlexDevice.h>

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <new>
#include <vector>

#include <sstream>
#include <iostream>

#include "../core/maths.h"
#include "../core/types.h"
#include "../core/platform.h"
#include "../core/mesh.h"
#include "../core/voxelize.h"
#include "../core/sdf.h"
#include "../core/pfm.h"
#include "../core/tga.h"
#include "../core/perlin.h"
#include "../core/convex.h"

#include "printer.h"
#include "./opengl/shadersGL.h"

struct EGLInternalData2 {
    bool m_isInitialized;

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;

    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;

    EGLInternalData2()
    : m_isInitialized(false),
    m_windowWidth(0),
    m_windowHeight(0) {}
};

bool g_Error = false;

void ErrorCallback(NvFlexErrorSeverity severity, const char* msg, const char* file, int line) {
	printf("Flex: %s - %s:%d\n", msg, file, line);
	g_Error = (severity == eNvFlexLogError);
}

class Pyflex {
	EGLInternalData2* m_data = NULL;
	NvFlexSolverDesc g_solverDesc;
	NvFlexSolver* g_solver = NULL;
	NvFlexLibrary* g_flexLib = NULL;
	NvFlexParams g_params;
	Printer *printer = NULL;
	GpuTimers g_GpuTimers;

	Emitter g_e;
	Mesh* g_mesh = NULL;
	SimBuffers* g_buffers = NULL;
	OpenGL::FluidRenderBuffers* g_fluidRenderBuffers;
	OpenGL::DiffuseRenderBuffers* g_diffuseRenderBuffers = NULL;
	std::map<NvFlexTriangleMeshId, OpenGL::GpuMesh*> g_meshes;
	std::map<NvFlexConvexMeshId, OpenGL::GpuMesh*> g_convexes;
	Vec3 g_clearColor = Vec3(0.f);

	OpenGL::MSAABuffers g_msaa;
	OpenGL::FluidRenderer* g_fluidRenderer = NULL;
	OpenGL::ShadowMap* g_shadowMap = NULL;
	GLuint s_diffuseProgram = GLuint(-1);

	Vec4 view_bounds = Vec4(-9.f, 9.f, -9.f, 9.f);

	#pragma region Constants
	int g_frame = 0;
	bool g_pause = false;

	float g_dt = 1.0f / 60.0f;
	float g_waveTime = 0.0f;
	float g_windTime = 0.0f;
	float g_windStrength = 0.0f;

	float g_blur;
	Vec4 g_fluidColor;
	Vec3 g_meshColor;
	bool g_drawEllipsoids;
	bool g_drawPoints;
	bool g_drawCloth;
	float g_expandCloth;

	bool g_drawOpaque;
	int g_drawSprings;
	bool g_drawDiffuse;
	bool g_drawMesh;
	bool g_drawRopes;
	bool g_drawDensity = false;
	float g_ior;
	float g_lightDistance;
	float g_fogDistance;

	float g_camSpeed;
	float g_camNear;
	float g_camFar;

	float g_pointScale;
	float g_ropeScale;
	float g_drawPlaneBias;

	int g_numSubsteps;

	float g_diffuseScale;
	Vec4 g_diffuseColor;
	float g_diffuseMotionScale;
	bool g_diffuseShadow;
	float g_diffuseInscatter;
	float g_diffuseOutscatter;

	int g_numSolidParticles = 0;

	float g_waveFrequency = 1.5f;
	float g_waveAmplitude = 1.0f;
	float g_waveFloorTilt = 0.0f;
	bool g_emit = false;
	bool g_warmup = false;

	int g_mouseParticle = -1;

	int g_maxDiffuseParticles;
	int g_maxNeighborsPerParticle;
	int g_numExtraParticles;
	int g_maxContactsPerParticle;

	Vec3 g_sceneLower;
	Vec3 g_sceneUpper;

	float g_wavePlane;

	bool g_interop = true;

	float g_spotMin = 0.5f;
	float g_spotMax = 1.0f;

	Vec3 g_lightPos;
	Vec3 g_lightDir;
	Vec3 g_lightTarget;
	#pragma endregion

	// variables
	int g_screenWidth;
	int g_screenHeight;
	public:
	
	~Pyflex() {
		cleanup();
		glDeleteProgram(g_fluidRenderer->mEllipsoidDepthProgram);
		delete g_fluidRenderer;
		eglDestroySurface(m_data->egl_display, m_data->egl_surface);
		delete m_data;
	}

	void init(int width, int height, float min_x, float max_x, float min_y, float max_y, int renderDevice) {
		view_bounds = Vec4(min_x, max_x, min_y, max_y);
		this->initGL(width, height, renderDevice);
		this->initFleX(renderDevice);
		this->initRendering();
	}

	void initGL(int width, int height, int m_renderDevice) {
		this->g_screenWidth = width;
		this->g_screenHeight = height;
		
		EGLBoolean success;
		EGLint num_configs;
		EGLConfig egl_config;
		EGLSurface egl_surface;
		EGLContext egl_context;
		EGLDisplay egl_display;

		EGLint egl_config_attribs[] = {
			EGL_RED_SIZE,
			8,
			EGL_GREEN_SIZE,
			8,
			EGL_BLUE_SIZE,
			8,
			EGL_DEPTH_SIZE,
			8,
			EGL_SURFACE_TYPE,
			EGL_PBUFFER_BIT,
			EGL_RENDERABLE_TYPE,
			EGL_OPENGL_BIT,
			EGL_NONE
		};
		
		EGLint egl_pbuffer_attribs[] = {
			EGL_WIDTH, g_screenWidth, EGL_HEIGHT, g_screenHeight,
			EGL_NONE,
		};
		
		this->m_data = new EGLInternalData2();
		m_data->m_renderDevice = m_renderDevice;

		// Load EGL functions
		int egl_version = gladLoaderLoadEGL(NULL);
		if(!egl_version) {
			fprintf(stderr, "failed to EGL with glad.\n");
			exit(EXIT_FAILURE);
		};

		// Query EGL Devices
		const int max_devices = 32;
		EGLDeviceEXT egl_devices[max_devices];
		EGLint num_devices = 0;
		EGLint egl_error = eglGetError();
		if (!eglQueryDevicesEXT(max_devices, egl_devices, &num_devices) || egl_error != EGL_SUCCESS) {
			printf("eglQueryDevicesEXT Failed.\n");
			m_data->egl_display = EGL_NO_DISPLAY;
		}

		// Query EGL Screens
		if(m_data->m_renderDevice == -1) {
			printf("devices = %d\n", num_devices);
			// Chose default screen, by trying all
			for (EGLint i = 0; i < num_devices; ++i) {
				// Set display
				EGLDisplay display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, egl_devices[i], NULL);
				if (eglGetError() == EGL_SUCCESS && display != EGL_NO_DISPLAY) {
					int major, minor;
					EGLBoolean initialized = eglInitialize(display, &major, &minor);
					if (eglGetError() == EGL_SUCCESS && initialized == EGL_TRUE) {
						m_data->egl_display = display;
					}
				}
			}
		} else {
			// Chose specific screen, by using m_renderDevice
			if (m_data->m_renderDevice < 0 || m_data->m_renderDevice >= num_devices) {
				fprintf(stderr, "Invalid render_device choice: %d < %d.\n", m_data->m_renderDevice, num_devices);
				exit(EXIT_FAILURE);
			}

			// Set display
			EGLDisplay display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, egl_devices[m_data->m_renderDevice], NULL);
			if (eglGetError() == EGL_SUCCESS && display != EGL_NO_DISPLAY) {
				int major, minor;
				EGLBoolean initialized = eglInitialize(display, &major, &minor);
				if (eglGetError() == EGL_SUCCESS && initialized == EGL_TRUE) {
					m_data->egl_display = display;
				}
			}
		}

		if (!eglInitialize(m_data->egl_display, NULL, NULL)) {
			fprintf(stderr, "Unable to initialize EGL\n");
			exit(EXIT_FAILURE);
		}

		egl_version = gladLoaderLoadEGL(m_data->egl_display);
		if (!egl_version) {
			fprintf(stderr, "Unable to reload EGL.\n");
			exit(EXIT_FAILURE);
		}
		// printf("Loaded EGL %d.%d after reload.\n", GLAD_VERSION_MAJOR(egl_version), GLAD_VERSION_MINOR(egl_version));


		m_data->success = eglBindAPI(EGL_OPENGL_API);
		if (!m_data->success) {
			fprintf(stderr, "Failed to bind OpenGL API.\n");
			exit(EXIT_FAILURE);
		}

		m_data->success = eglChooseConfig(m_data->egl_display, egl_config_attribs, &m_data->egl_config, 1, &m_data->num_configs);
		if (!m_data->success) {
			fprintf(stderr, "Failed to choose config (eglError: %d)\n", eglGetError());
			exit(EXIT_FAILURE);
		}
		if (m_data->num_configs != 1) {
			fprintf(stderr, "Didn't get exactly one config, but %d\n", m_data->num_configs);
			exit(EXIT_FAILURE);
		}

		m_data->egl_surface = eglCreatePbufferSurface(m_data->egl_display, m_data->egl_config, egl_pbuffer_attribs);
		if (m_data->egl_surface == EGL_NO_SURFACE) {
			fprintf(stderr, "Unable to create EGL surface (eglError: %d)\n", eglGetError());
			exit(EXIT_FAILURE);
		}

		m_data->egl_context = eglCreateContext(m_data->egl_display, m_data->egl_config, EGL_NO_CONTEXT, NULL);
		if (!m_data->egl_context) {
			fprintf(stderr, "Unable to create EGL context (eglError: %d)\n",eglGetError());
			exit(EXIT_FAILURE);
		}

		m_data->success = eglMakeCurrent(m_data->egl_display, m_data->egl_surface, m_data->egl_surface, m_data->egl_context);
		if (!m_data->success) {
			fprintf(stderr, "Failed to make context current (eglError: %d)\n", eglGetError());
			exit(EXIT_FAILURE);
		}

		if (!gladLoadGLLoader(eglGetProcAddress)) {
			fprintf(stderr, "failed to load GL with glad.\n");
			exit(EXIT_FAILURE);
		}

		const GLubyte* ven = glGetString(GL_VENDOR);
		// printf("GL_VENDOR=%s\n", ven);
		const GLubyte* ren = glGetString(GL_RENDERER);
		// printf("GL_RENDERER=%s\n", ren);
		const GLubyte* ver = glGetString(GL_VERSION);
		printf("GL_VERSION=%s\n", ver);
		const GLubyte* sl = glGetString(GL_SHADING_LANGUAGE_VERSION);
		// printf("GL_SHADING_LANGUAGE_VERSION=%s\n", sl);
	}

	void initFleX(int renderDevice) {
		int g_device = renderDevice;
		char g_deviceName[256];
		bool g_extensions = true;

		NvFlexTimers g_timers;
		int g_numDetailTimers;
		NvFlexDetailTimer* g_detailTimers;

		NvFlexInitDesc desc;
		desc.deviceIndex = g_device;
		desc.enableExtensions = g_extensions;
		desc.renderDevice = 0;
		desc.renderContext = 0;
		desc.computeContext = 0;
		desc.computeType = eNvFlexCUDA;

		this->g_flexLib = NvFlexInit(NV_FLEX_VERSION, ErrorCallback, &desc);
		if (g_Error || g_flexLib == nullptr) {
			printf("Could not initialize Flex, exiting.\n");
			exit(-1);
		}
		// store device name
		strcpy(g_deviceName, NvFlexGetDeviceName(g_flexLib));
		printf("Compute Device: %s\n", g_deviceName);
		// prepare scenes
		printer = new Printer(g_flexLib);
	}

	void initRendering() {
		g_fluidRenderer = OpenGL::CreateFluidRenderer(g_msaa.g_msaaFbo, g_screenWidth, g_screenHeight, true);
	}

	void cleanup() {
		if (g_solver) {
			if (g_buffers) {
				DestroyBuffers(g_buffers);
			}

			OpenGL::DestroyFluidRenderBuffers(g_fluidRenderBuffers);

			for (auto& iter : g_meshes) {
				NvFlexDestroyTriangleMesh(g_flexLib, iter.first);
				OpenGL::DestroyGpuMesh(iter.second);
			}

			for (auto& iter : g_convexes) {
				NvFlexDestroyConvexMesh(g_flexLib, iter.first);
				OpenGL::DestroyGpuMesh(iter.second);
			}

			g_meshes.clear();
			g_convexes.clear();

			NvFlexDestroySolver(g_solver);
			g_solver = nullptr;
		}
	}

	void reset(int numSubsteps, float materialViscosity, int materialReservoir, int meshID) {
		cleanup();
		#pragma region Create buffers
		// alloc buffers
		g_buffers = AllocBuffers(g_flexLib);
		// map during initialization
		MapBuffers(g_buffers);

		g_buffers->positions.resize(0);
		g_buffers->velocities.resize(0);
		g_buffers->phases.resize(0);

		g_buffers->rigidOffsets.resize(0);
		g_buffers->rigidIndices.resize(0);
		g_buffers->rigidMeshSize.resize(0);
		g_buffers->rigidRotations.resize(0);
		g_buffers->rigidTranslations.resize(0);
		g_buffers->rigidCoefficients.resize(0);
		g_buffers->rigidPlasticThresholds.resize(0);
		g_buffers->rigidPlasticCreeps.resize(0);
		g_buffers->rigidLocalPositions.resize(0);
		g_buffers->rigidLocalNormals.resize(0);

		g_buffers->springIndices.resize(0);
		g_buffers->springLengths.resize(0);
		g_buffers->springStiffness.resize(0);
		g_buffers->triangles.resize(0);
		g_buffers->triangleNormals.resize(0);
		g_buffers->uvs.resize(0);

		g_buffers->shapeGeometry.resize(0);
		g_buffers->shapePositions.resize(0);
		g_buffers->shapeRotations.resize(0);
		g_buffers->shapePrevPositions.resize(0);
		g_buffers->shapePrevRotations.resize(0);
		g_buffers->shapeFlags.resize(0);
		#pragma endregion
		#pragma region Set Constants
		delete g_mesh; g_mesh = NULL;

		g_frame = 0;
		g_pause = false;

		g_dt = 1.0f / 60.0f;
		g_waveTime = 0.0f;
		g_windTime = 0.0f;
		g_windStrength = 1.0f;

		g_blur = 1.0f;
		g_fluidColor = Vec4(0.1f, 0.4f, 0.8f, 1.0f);
		g_meshColor = Vec3(0.9f, 0.9f, 0.9f);
		g_drawEllipsoids = false;
		g_drawPoints = true;
		g_drawCloth = true;
		g_expandCloth = 0.0f;

		g_drawOpaque = false;
		g_drawSprings = false;
		g_drawDiffuse = false;
		g_drawMesh = true;
		g_drawRopes = true;
		g_drawDensity = false;
		g_ior = 1.0f;
		g_lightDistance = 2.0f;
		g_fogDistance = 0.005f;

		g_camSpeed = 0.075f;
		g_camNear = 0.01f;
		g_camFar = 1000.0f;

		g_pointScale = 1.0f;
		g_ropeScale = 1.0f;
		g_drawPlaneBias = 0.0f;
		#pragma endregion
		#pragma region g_params
		g_params.gravity[0] = 0.0f;
		g_params.gravity[1] = -9.8f;
		g_params.gravity[2] = 0.0f;

		g_params.wind[0] = 0.0f;
		g_params.wind[1] = 0.0f;
		g_params.wind[2] = 0.0f;

		g_params.radius = 0.15f;
		g_params.viscosity = 0.0f;
		g_params.dynamicFriction = 0.0f;
		g_params.staticFriction = 0.0f;
		g_params.particleFriction = 0.0f; // scale friction between particles by default
		g_params.freeSurfaceDrag = 0.0f;
		g_params.drag = 0.0f;
		g_params.lift = 0.0f;
		g_params.numIterations = 3;
		g_params.fluidRestDistance = 0.0f;
		g_params.solidRestDistance = 0.0f;

		g_params.anisotropyScale = 1.0f;
		g_params.anisotropyMin = 0.1f;
		g_params.anisotropyMax = 2.0f;
		g_params.smoothing = 1.0f;

		g_params.dissipation = 0.0f;
		g_params.damping = 0.0f;
		g_params.particleCollisionMargin = 0.0f;
		g_params.shapeCollisionMargin = 0.0f;
		g_params.collisionDistance = 0.0f;
		g_params.sleepThreshold = 0.0f;
		g_params.shockPropagation = 0.0f;
		g_params.restitution = 0.0f;

		g_params.maxSpeed = FLT_MAX;
		g_params.maxAcceleration = 100.0f;    // approximately 10x gravity

		g_params.relaxationMode = eNvFlexRelaxationLocal;
		g_params.relaxationFactor = 1.0f;
		g_params.solidPressure = 1.0f;
		g_params.adhesion = 0.0f;
		g_params.cohesion = 0.025f;
		g_params.surfaceTension = 0.0f;
		g_params.vorticityConfinement = 0.0f;
		g_params.buoyancy = 1.0f;
		g_params.diffuseThreshold = 100.0f;
		g_params.diffuseBuoyancy = 1.0f;
		g_params.diffuseDrag = 0.8f;
		g_params.diffuseBallistic = 16;
		g_params.diffuseLifetime = 2.0f;

		g_numSubsteps = numSubsteps;
		#pragma endregion
		#pragma region Setup planes and particles
		// planes created after particles
		g_params.numPlanes = 1;

		g_diffuseScale = 0.5f;
		g_diffuseColor = 1.0f;
		g_diffuseMotionScale = 1.0f;
		g_diffuseShadow = false;
		g_diffuseInscatter = 0.8f;
		g_diffuseOutscatter = 0.53f;

		g_numSolidParticles = 0;

		g_waveFrequency = 1.5f;
		g_waveAmplitude = 1.5f;
		g_waveFloorTilt = 0.0f;
		g_emit = false;
		g_warmup = false;

		g_mouseParticle = -1;

		g_maxDiffuseParticles = 0;    // number of diffuse particles
		g_maxNeighborsPerParticle = 96;
		g_numExtraParticles = 0;    // number of particles allocated but not made active
		g_maxContactsPerParticle = 6;

		g_sceneLower = FLT_MAX;
		g_sceneUpper = -FLT_MAX;
		#pragma endregion
		#pragma region Init Solver
		// initialize solver desc
		NvFlexSetSolverDescDefaults(&g_solverDesc);
		g_e = printer->Initialize(g_buffers, g_meshes, g_params, materialViscosity, materialReservoir, meshID);
		g_sceneUpper.z = 5.0f;
		g_numExtraParticles = materialReservoir;
		g_lightDistance *= 2.5f;
		uint32_t numParticles = g_buffers->positions.size();
		uint32_t maxParticles = numParticles + g_numExtraParticles;

		if (g_params.solidRestDistance == 0.0f)
			g_params.solidRestDistance = g_params.radius;
		// if fluid present then we assume solid particles have the same radius
		if (g_params.fluidRestDistance > 0.0f)
			g_params.solidRestDistance = g_params.fluidRestDistance;
		// set collision distance automatically based on rest distance if not already set
		if (g_params.collisionDistance == 0.0f)
			g_params.collisionDistance = Max(g_params.solidRestDistance, g_params.fluidRestDistance)*0.5f;
		// default particle friction to 10% of shape friction
		if (g_params.particleFriction == 0.0f)
			g_params.particleFriction = g_params.dynamicFriction*0.1f;
		// add a margin for detecting contacts between particles and shapes
		if (g_params.shapeCollisionMargin == 0.0f)
			g_params.shapeCollisionMargin = g_params.collisionDistance*0.5f;
		#pragma endregion
		#pragma region Set bounds
		// calculate particle bounds
		Vec3 particleLower, particleUpper;
		GetParticleBounds(g_buffers, particleLower, particleUpper);

		// accommodate shapes
		Vec3 shapeLower, shapeUpper;
		GetShapeBounds(g_flexLib, g_buffers, shapeLower, shapeUpper);

		// update bounds
		g_sceneLower = Min(Min(g_sceneLower, particleLower), shapeLower);
		g_sceneUpper = Max(Max(g_sceneUpper, particleUpper), shapeUpper);

		g_sceneLower -= g_params.collisionDistance;
		g_sceneUpper += g_params.collisionDistance;
		#pragma endregion
		#pragma region Update collision planes
		// update collision planes to match flexs
		Vec3 up = Normalize(Vec3(-g_waveFloorTilt, 1.0f, 0.0f));

		(Vec4&)g_params.planes[0] = Vec4(up.x, up.y, up.z, 0.0f);
		(Vec4&)g_params.planes[1] = Vec4(0.0f, 0.0f, 1.0f, -g_sceneLower.z);
		(Vec4&)g_params.planes[2] = Vec4(1.0f, 0.0f, 0.0f, -g_sceneLower.x);
		(Vec4&)g_params.planes[3] = Vec4(-1.0f, 0.0f, 0.0f, g_sceneUpper.x);
		(Vec4&)g_params.planes[4] = Vec4(0.0f, 0.0f, -1.0f, g_sceneUpper.z);
		(Vec4&)g_params.planes[5] = Vec4(0.0f, -1.0f, 0.0f, g_sceneUpper.y);

		g_wavePlane = g_params.planes[2][3];

		g_buffers->diffusePositions.resize(g_maxDiffuseParticles);
		g_buffers->diffuseVelocities.resize(g_maxDiffuseParticles);
		g_buffers->diffuseCount.resize(1, 0);
		#pragma endregion
		#pragma region Laplacian positions for fluid rendering
		// for fluid rendering these are the Laplacian smoothed positions
		g_buffers->smoothPositions.resize(maxParticles);

		g_buffers->normals.resize(0);
		g_buffers->normals.resize(maxParticles);

		// initialize normals (just for rendering before simulation starts)
		int numTris = g_buffers->triangles.size() / 3;
		for (int i = 0; i < numTris; ++i) {
			Vec3 v0 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 0]]);
			Vec3 v1 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 1]]);
			Vec3 v2 = Vec3(g_buffers->positions[g_buffers->triangles[i * 3 + 2]]);

			Vec3 n = Cross(v1 - v0, v2 - v0);

			g_buffers->normals[g_buffers->triangles[i * 3 + 0]] += Vec4(n, 0.0f);
			g_buffers->normals[g_buffers->triangles[i * 3 + 1]] += Vec4(n, 0.0f);
			g_buffers->normals[g_buffers->triangles[i * 3 + 2]] += Vec4(n, 0.0f);
		}

		for (int i = 0; i < int(maxParticles); ++i)
			g_buffers->normals[i] = Vec4(SafeNormalize(Vec3(g_buffers->normals[i]), Vec3(0.0f, 1.0f, 0.0f)), 0.0f);
		#pragma endregion
		#pragma region Solver Init
		g_solverDesc.maxParticles = maxParticles;
		g_solverDesc.maxDiffuseParticles = g_maxDiffuseParticles;
		g_solverDesc.maxNeighborsPerParticle = g_maxNeighborsPerParticle;
		g_solverDesc.maxContactsPerParticle = g_maxContactsPerParticle;

		// main create method for the Flex solver
		g_solver = NvFlexCreateSolver(g_flexLib, &g_solverDesc);
		#pragma endregion
		#pragma region Particle setup
		// create active indices (just a contiguous block for the demo)
		g_buffers->activeIndices.resize(g_buffers->positions.size());
		for (int i = 0; i < g_buffers->activeIndices.size(); ++i)
			g_buffers->activeIndices[i] = i;

		// resize particle buffers to fit
		g_buffers->positions.resize(maxParticles);
		g_buffers->velocities.resize(maxParticles);
		g_buffers->phases.resize(maxParticles);

		g_buffers->densities.resize(maxParticles);
		g_buffers->anisotropy1.resize(maxParticles);
		g_buffers->anisotropy2.resize(maxParticles);
		g_buffers->anisotropy3.resize(maxParticles);

		// save rest positions
		g_buffers->restPositions.resize(g_buffers->positions.size());
		for (int i = 0; i < g_buffers->positions.size(); ++i)
			g_buffers->restPositions[i] = g_buffers->positions[i];
		#pragma endregion
		#pragma region Rigid constraints
		// builds rigids constraints
		if (g_buffers->rigidOffsets.size()) {
			assert(g_buffers->rigidOffsets.size() > 1);

			const int numRigids = g_buffers->rigidOffsets.size() - 1;

			// If the centers of mass for the rigids are not yet computed, this is done here
			// (If the CreateParticleShape method is used instead of the NvFlexExt methods, the centers of mass will be calculated here)
			if (g_buffers->rigidTranslations.size() == 0) {
				g_buffers->rigidTranslations.resize(g_buffers->rigidOffsets.size() - 1, Vec3());
				CalculateRigidCentersOfMass(&g_buffers->positions[0], g_buffers->positions.size(), &g_buffers->rigidOffsets[0], &g_buffers->rigidTranslations[0], &g_buffers->rigidIndices[0], numRigids);
			}

			// calculate local rest space positions
			g_buffers->rigidLocalPositions.resize(g_buffers->rigidOffsets.back());
			CalculateRigidLocalPositions(&g_buffers->positions[0], &g_buffers->rigidOffsets[0], &g_buffers->rigidTranslations[0], &g_buffers->rigidIndices[0], numRigids, &g_buffers->rigidLocalPositions[0]);

			// set rigidRotations to correct length, probably NULL up until here
			g_buffers->rigidRotations.resize(g_buffers->rigidOffsets.size() - 1, Quat());
		}
		#pragma endregion
		// unmap so we can start transferring data to GPU
		UnmapBuffers(g_buffers);
		#pragma region Send data to// Send data to Flex
		NvFlexCopyDesc copyDesc;
		copyDesc.dstOffset = 0;
		copyDesc.srcOffset = 0;
		copyDesc.elementCount = numParticles;

		NvFlexSetParams(g_solver, &g_params);
		NvFlexSetParticles(g_solver, g_buffers->positions.buffer, &copyDesc);
		NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, &copyDesc);
		NvFlexSetNormals(g_solver, g_buffers->normals.buffer, &copyDesc);
		NvFlexSetPhases(g_solver, g_buffers->phases.buffer, &copyDesc);
		NvFlexSetRestParticles(g_solver, g_buffers->restPositions.buffer, &copyDesc);

		NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, &copyDesc);
		NvFlexSetActiveCount(g_solver, numParticles);

		// springs
		if (g_buffers->springIndices.size()) {
			assert((g_buffers->springIndices.size() & 1) == 0);
			assert((g_buffers->springIndices.size() / 2) == g_buffers->springLengths.size());

			NvFlexSetSprings(g_solver, g_buffers->springIndices.buffer, g_buffers->springLengths.buffer, g_buffers->springStiffness.buffer, g_buffers->springLengths.size());
		}

		// rigids
		if (g_buffers->rigidOffsets.size()) {
			NvFlexSetRigids(g_solver, g_buffers->rigidOffsets.buffer, g_buffers->rigidIndices.buffer, g_buffers->rigidLocalPositions.buffer, g_buffers->rigidLocalNormals.buffer, g_buffers->rigidCoefficients.buffer, g_buffers->rigidPlasticThresholds.buffer, g_buffers->rigidPlasticCreeps.buffer, g_buffers->rigidRotations.buffer, g_buffers->rigidTranslations.buffer, g_buffers->rigidOffsets.size() - 1, g_buffers->rigidIndices.size());
		}

		// inflatables
		if (g_buffers->inflatableTriOffsets.size()) {
			NvFlexSetInflatables(g_solver, g_buffers->inflatableTriOffsets.buffer, g_buffers->inflatableTriCounts.buffer, g_buffers->inflatableVolumes.buffer, g_buffers->inflatablePressures.buffer, g_buffers->inflatableCoefficients.buffer, g_buffers->inflatableTriOffsets.size());
		}

		// dynamic triangles
		if (g_buffers->triangles.size()) {
			NvFlexSetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer, g_buffers->triangles.size() / 3);
		}

		// collision shapes
		if (g_buffers->shapeFlags.size()) {
			NvFlexSetShapes(
				g_solver,
				g_buffers->shapeGeometry.buffer,
				g_buffers->shapePositions.buffer,
				g_buffers->shapeRotations.buffer,
				g_buffers->shapePrevPositions.buffer,
				g_buffers->shapePrevRotations.buffer,
				g_buffers->shapeFlags.buffer,
				int(g_buffers->shapeFlags.size()));
		}
		#pragma endregion
		// create render buffers
		g_fluidRenderBuffers = OpenGL::CreateFluidRenderBuffers(g_flexLib, maxParticles, g_interop);
	}

	void UpdateEmitters() {
		int activeCount = NvFlexGetActiveCount(g_solver);
		if (!g_e.mEnabled)
			return;

		Vec3 emitterDir = g_e.mDir;
		Vec3 emitterRight = g_e.mRight;
		Vec3 emitterPos = g_e.mPos;
		Vec3 emitterVelocity = g_e.mVelocity;

		float r = g_params.fluidRestDistance;
		int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);

		float numParticles = (g_e.mFlow / r)*g_dt;

		// whole number to emit
		auto n = int(numParticles + g_e.mLeftOver);

		if (n)
			g_e.mLeftOver = (numParticles + g_e.mLeftOver) - n;
		else
			g_e.mLeftOver += numParticles;

		// create a grid of particles (n particles thick)
		for (int k = 0; k < n; ++k) {
			int emitterWidth = g_e.mWidth;
			int numParticles = emitterWidth*emitterWidth;
			for (int i = 0; i < numParticles; ++i) {
				float x = float(i%emitterWidth) - float(emitterWidth / 2);
				float y = float((i / emitterWidth) % emitterWidth) - float(emitterWidth / 2);

				if ((sqr(x) + sqr(y)) <= (emitterWidth / 2)*(emitterWidth / 2)) {
					Vec3 up = Normalize(Cross(emitterDir, emitterRight));
					Vec3 offset = r*(emitterRight*x + up*y) + float(k)*emitterDir*r;

					if (activeCount < g_buffers->positions.size()) {
						g_buffers->positions[activeCount] = Vec4(emitterPos + offset, 1.0f);
						g_buffers->velocities[activeCount] = emitterDir*g_e.mFlow * 2.f + emitterVelocity;
						g_buffers->phases[activeCount] = phase;

						g_buffers->activeIndices.push_back(activeCount);

						activeCount++;
					}
				}
			}
		}
	}

	void RenderScene() {
		int numParticles = NvFlexGetActiveCount(g_solver);
		int numDiffuse = g_buffers->diffuseCount[0];

		//---------------------------------------------------
		// use VBO buffer wrappers to allow Flex to write directly to the OpenGL buffers
		// Flex will take care of any CUDA interop mapping/unmapping during the get() operations

		if (numParticles) {
			if (g_interop) {
				// copy data directly from solver to the renderer buffers
				OpenGL::UpdateFluidRenderBuffers(g_fluidRenderBuffers, g_solver, g_drawEllipsoids, g_drawDensity);
			} else {
				OpenGL::UpdateFluidRenderBuffers(g_fluidRenderBuffers,
					&g_buffers->smoothPositions[0],
					(g_drawDensity) ? &g_buffers->densities[0] : (float*)&g_buffers->phases[0],
					&g_buffers->anisotropy1[0],
					&g_buffers->anisotropy2[0],
					&g_buffers->anisotropy3[0],
					g_buffers->positions.size(),
					&g_buffers->activeIndices[0],
					numParticles);
			}
		}

		//---------------------------------------
		// setup view and state

		float fov = kPi / 4.0f;
		float aspect = float(g_screenWidth) / g_screenHeight;

		//------------------------------------
		// lighting pass

		// expand scene bounds to fit most scenes
		g_sceneLower = Min(g_sceneLower, Vec3(-2.0f, 0.0f, -2.0f));
		g_sceneUpper = Max(g_sceneUpper, Vec3(2.0f, 2.0f, 2.0f));

		Vec3 sceneExtents = g_sceneUpper - g_sceneLower;
		Vec3 sceneCenter = 0.5f*(g_sceneUpper + g_sceneLower);

		g_lightDir = Normalize(Vec3(5.0f, 15.0f, 7.5f));
		g_lightPos = sceneCenter + g_lightDir*Length(sceneExtents)*g_lightDistance;
		g_lightTarget = sceneCenter;

		// calculate tight bounds for shadow frustum
		float lightFov = 2.0f*atanf(Length(g_sceneUpper - sceneCenter) / Length(g_lightPos - sceneCenter));

		// scale and clamp fov for aesthetics
		lightFov = Clamp(lightFov, DegToRad(25.0f), DegToRad(65.0f));

		Matrix44 lightPerspective = ProjectionMatrix(RadToDeg(lightFov), 1.0f, 1.0f, 1000.0f);
		Matrix44 lightView = LookAtMatrix(Point3(g_lightPos), Point3(g_lightTarget));
		Matrix44 lightTransform = lightPerspective*lightView;

		// radius used for drawing
		float radius = Max(g_params.solidRestDistance, g_params.fluidRestDistance)*0.5f*g_pointScale;

		//----------------
		// lighting pass

		OpenGL::SetCullMode(true);
		OpenGL::RenderEllipsoids(g_buffers, g_meshes, g_convexes, g_msaa.g_msaaFbo, g_mesh, g_fluidRenderer, g_fluidRenderBuffers, NULL, view_bounds, g_spotMin, g_spotMax, numParticles - g_numSolidParticles, g_numSolidParticles, radius, float(g_screenWidth), aspect, fov, g_lightPos, g_lightTarget, lightTransform, g_shadowMap, g_fluidColor, g_blur, g_ior, false);
	}

	void step(float x, float y, float z, float flow, double *frame) {
		#pragma region Scene update
		// Scene Update
		MapBuffers(g_buffers);
		// Getting timers causes CPU/GPU sync, so we do it after a map
		float newSimLatency = NvFlexGetDeviceLatency(g_solver, &g_GpuTimers.computeBegin, &g_GpuTimers.computeEnd, &g_GpuTimers.computeFreq);

		Vec4 updateVector(x, y, z, flow);
		printer->Update(g_buffers, g_e, updateVector, g_dt);
		UpdateEmitters();
		#pragma endregion
		#pragma region Render
		// Render
		OpenGL::StartFrame(g_msaa.g_msaaFbo, Vec4(g_clearColor, 1.0f));
		// main scene render
		RenderScene();
		OpenGL::EndFrame(g_msaa.g_msaaFbo, g_screenWidth, g_screenHeight);
		UnmapBuffers(g_buffers);
		#pragma endregion
		#pragma region Flex update
		// // Flex Update
		// send any particle updates to the solver
		NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
		NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
		NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
		NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, nullptr);

		NvFlexSetActiveCount(g_solver, g_buffers->activeIndices.size());

		NvFlexSetShapes(
			g_solver,
			g_buffers->shapeGeometry.buffer,
			g_buffers->shapePositions.buffer,
			g_buffers->shapeRotations.buffer,
			g_buffers->shapePrevPositions.buffer,
			g_buffers->shapePrevRotations.buffer,
			g_buffers->shapeFlags.buffer,
			int(g_buffers->shapeFlags.size()));

		NvFlexSetParams(g_solver, &g_params);
		NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, false);

		g_frame++;

		// read back base particle data
		// Note that flexGet calls don't wait for the GPU, they just queue a GPU copy
		// to be executed later.
		// When we're ready to read the fetched buffers we'll Map them, and that's when
		// the CPU will wait for the GPU flex update and GPU copy to finish.
		NvFlexGetParticles(g_solver, g_buffers->positions.buffer, nullptr);
		NvFlexGetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
		NvFlexGetNormals(g_solver, g_buffers->normals.buffer, nullptr);

		// readback triangle normals
		if (g_buffers->triangles.size())
			NvFlexGetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer, g_buffers->triangles.size() / 3);

		// readback rigid transforms
		if (g_buffers->rigidOffsets.size())
			NvFlexGetRigids(g_solver, g_buffers->rigidOffsets.buffer, g_buffers->rigidIndices.buffer, g_buffers->rigidLocalPositions.buffer, nullptr, nullptr, nullptr, nullptr, g_buffers->rigidRotations.buffer, g_buffers->rigidTranslations.buffer);

		if (!g_interop) {
			NvFlexGetSmoothParticles(g_solver, g_buffers->smoothPositions.buffer, nullptr);
			NvFlexGetAnisotropy(g_solver, g_buffers->anisotropy1.buffer, g_buffers->anisotropy2.buffer, g_buffers->anisotropy3.buffer, NULL);
		}

		NvFlexGetDiffuseParticles(g_solver, nullptr, nullptr, g_buffers->diffuseCount.buffer);
		#pragma endregion
		#pragma region Present frame
		eglSwapInterval(m_data->egl_display, 1);
		eglSwapBuffers(m_data->egl_display, m_data->egl_surface);
		#pragma endregion
		#pragma region Copy frame
		int numData = g_screenWidth * g_screenHeight;
		float *data = new float[numData];
		glVerify(glReadPixels(0, 0, g_screenWidth, g_screenHeight, GL_RED, GL_FLOAT, data));
		for (int i = 0; i < numData; i++) {
			frame[i] = data[i];
		}
		delete[] data;
		#pragma endregion
	}

	float _cc = 0.0;

	void test(double *frame) {	
		float vertices[] = {
			-0.5f+0.5*_cc, -0.5f, 0.0f,
			0.5f+0.5*_cc, -0.5f, 0.0f,
			0.0f+0.5*_cc,  0.5f, 0.0f
		};

		unsigned int VBO;
		glGenBuffers(1, &VBO);

		glBindBuffer(GL_ARRAY_BUFFER, VBO);  
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		const char *vertexShaderSource = "#version 460 core\nlayout (location = 0) in vec3 aPos;\nvoid main()\n{\ngl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n}";

		unsigned int vertexShader;
		vertexShader = glCreateShader(GL_VERTEX_SHADER);

		glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
		glCompileShader(vertexShader);

		int  success;
		char infoLog[512];
		glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
		if(!success) {
			glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		std::stringstream ss;
		ss << "#version 460 core\nout vec4 FragColor;\nvoid main()\n{\nFragColor = vec4(" << _cc+0.5 << ", 0.5f, 0.2f, 1.0f);\n}";
		std::string fragmentShaderSource_str = ss.str();
		const char *fragmentShaderSource = fragmentShaderSource_str.c_str();
		_cc += 0.01;

		unsigned int fragmentShader;
		fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
		glCompileShader(fragmentShader);

		unsigned int shaderProgram;
		shaderProgram = glCreateProgram();

		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glLinkProgram(shaderProgram);

		glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
		if(!success) {
			glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
			printf("fuck\n");
		}

		glUseProgram(shaderProgram);
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
		
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);  

		unsigned int VAO;
		glGenVertexArrays(1, &VAO); 
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); 
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0); 

		// 
		float cc = 0.0;
		glClearColor(cc, cc, cc, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		eglSwapInterval(m_data->egl_display, 1);
		eglSwapBuffers(m_data->egl_display, m_data->egl_surface);

		int numData = g_screenWidth*g_screenHeight;
		float *data = new float[numData];
		glVerify(glReadPixels(0, 0, g_screenWidth, g_screenHeight, GL_RED, GL_FLOAT, data));
		for (int i = 0; i < numData; i++) {
			frame[i] = data[i];
		}
		delete[] data;
	}

	void test2(double *frame) {
		// vertices
		float vertices[] = {
			-0.5f+0.5*_cc, -0.5f, 0.0f,
			0.5f+0.5*_cc, -0.5f, 0.0f,
			0.0f+0.5*_cc,  0.5f, 0.0f
		};
		_cc += 0.1;
		float quad[] = {
			-1.f, -1.f,
			 1.f, -1.f,
			-1.f,  1.f,
			 1.f, -1.f,
			 1.f,  1.f,
			-1.f,  1.f,
		};
		// create framebuffer
		unsigned int framebuffer;
		glGenFramebuffers(1, &framebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);   
		// generate texture
		unsigned int texColorBuffer;
		glGenTextures(1, &texColorBuffer);
		glBindTexture(GL_TEXTURE_2D, texColorBuffer);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
		// attach it to currently bound framebuffer object
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texColorBuffer, 0);

		unsigned int rbo;
		glGenRenderbuffers(1, &rbo);
		glBindRenderbuffer(GL_RENDERBUFFER, rbo); 
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 512, 512);  
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);  

		// first pass
		const char *vertexShaderSource =
			"#version 460 core\n"
			"layout (location = 0) in vec3 aPos;\n"
			"void main() {\n"
				"gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
			"}";

		unsigned int vertexShader;
		vertexShader = glCreateShader(GL_VERTEX_SHADER);

		glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
		glCompileShader(vertexShader);

		const char *fragmentShaderSource =
			"#version 460 core\n"
			"out vec4 FragColor;\n"
			"void main() {\n"
				"FragColor = vec4(1.f, 0.5f, 0.2f, 1.0f);\n"
			"}";

		unsigned int fragmentShader;
		fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
		glCompileShader(fragmentShader);

		unsigned int shaderProgram;
		shaderProgram = glCreateProgram();

		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glLinkProgram(shaderProgram);

		unsigned int VBO;
		glGenBuffers(1, &VBO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);  
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		unsigned int VAO;
		glGenVertexArrays(1, &VAO); 
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); 
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0); 

		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		
		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		// second pass

		const char *vertSource =
			"#version 460 core\n"
			"layout (location = 0) in vec2 aPos;\n"
			"out vec2 TexCoords;\n"
			"void main() {\n"
				"gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);\n"
				"TexCoords = (aPos+1.f)/2.f;\n"
			"}";

		unsigned int vertShader;
		vertShader = glCreateShader(GL_VERTEX_SHADER);

		glShaderSource(vertShader, 1, &vertSource, NULL);
		glCompileShader(vertShader);

		const char *fragSource =
			"#version 460 core\n"
			"out vec4 FragColor;\n"
			"in vec2 TexCoords;\n"
			"uniform sampler2D screenTexture;\n"
			"void main() {\n"
				"FragColor = texture(screenTexture, TexCoords);\n"
				"FragColor = 1.f - FragColor;\n"
			"}";

		unsigned int fragShader;
		fragShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragShader, 1, &fragSource, NULL);
		glCompileShader(fragShader);

		unsigned int shaderProgram2;
		shaderProgram2 = glCreateProgram();

		glAttachShader(shaderProgram2, vertShader);
		glAttachShader(shaderProgram2, fragShader);
		glLinkProgram(shaderProgram2);

		unsigned int quadVBO;
		glGenBuffers(1, &quadVBO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);  
		glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

		unsigned int quadVAO;
		glGenVertexArrays(1, &quadVAO); 
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW); 
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0); 
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0); // back to default
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f); 
		glClear(GL_COLOR_BUFFER_BIT);
		
		glUseProgram(shaderProgram2);
		glBindVertexArray(quadVAO);
		glDisable(GL_DEPTH_TEST);
		glBindTexture(GL_TEXTURE_2D, texColorBuffer);
		glDrawArrays(GL_TRIANGLES, 0, 6);  

		// finish and grab frame
		eglSwapInterval(m_data->egl_display, 1);
		eglSwapBuffers(m_data->egl_display, m_data->egl_surface);

		int numData = g_screenWidth*g_screenHeight;
		float *data = new float[numData];
		glVerify(glReadPixels(0, 0, g_screenWidth, g_screenHeight, GL_RED, GL_FLOAT, data));
		for (int i = 0; i < numData; i++) {
			frame[i] = data[i];
		}
		delete[] data;
	}

};

extern "C"  //Tells the compile to use C-linkage for the next scope.
{
    //Note: The interface this linkage region needs to use C only.  
    Pyflex* CreatePyflexInstance( void ) {
        // Note: Inside the function body, I can use C++. 
        return new Pyflex;
    }

    // TODO: why is it crashing at the end?
    void DeletePyflexInstance(Pyflex *ptr) {
         delete ptr; 
    }

    void InitPyflexInstance(Pyflex *ptr, int width, int height, float min_x, float max_x, float min_y, float max_y, int renderDevice) { 
		ptr->init(width, height, min_x, max_x, min_y, max_y, renderDevice);
    }

	void ResetPyflexInstance(Pyflex *ptr, int numSubsteps, float materialViscosity, int materialReservoir, int meshID) {
		ptr->reset(numSubsteps, materialViscosity, materialReservoir, meshID);
	}

	void StepPyflexInstance(Pyflex *ptr, float x, float y, float z, float flow, double *frame) {
		ptr->step(x, y, z, flow, frame);
	}

	void TestPyflexInstance(Pyflex *ptr, double *frame) {
		ptr->test2(frame);
	}
} //End C linkage scope.