Shader "Custom/PathTracer"
{
	Properties{
		_Color ("Color", Color) = (0,0,0,1)
		_Index ("Refractive Index", Float) = 1
		_Brightness ("Brightness", Float) = 0
		_Skybox ("Skybox", CUBE) = "white" {}
		_Noise ("Noise", 2D) = "white" {}
		_Depth ("Depth", Range (1, 20)) = 5
		_SampleCount ("Sample Count", Range (1, 50)) = 5
		_Aperture ("Aperture Size", Float) = 0.1
		_FocalDepth ("Focal Depth", Float) = 10
		[MaterialToggle] _AutoFocus("Auto Focus", Float) = 0
		_ApertureEdges ("Aperture Edge Count", Range (3, 100)) = 7
		_B ("B Coefficients", Vector) = (1.03961212, 0.231792344, 1.01046945, 0)
		_C ("C Coefficients", Vector) = (0.00600069867, 0.0200179144, 103.560653, 0)
		_Gamma ("Gamma", Float) = 1.0
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"

			#define M_PI 3.1415926535897932384626433832795
			#define RANDOM_IA 16807
			#define RANDOM_IM 2147483647
			#define RANDOM_AM (1.0/float(RANDOM_IM))
			#define RANDOM_IQ 127773u
			#define RANDOM_IR 2836
			#define RANDOM_MASK 123459876

			float PHI = 1.61803398874989484820459 * 00000.1; // Golden Ratio   
			float PI  = 3.14159265358979323846264 * 00000.1; // PI
			float SRT = 1.41421356237309504880169 * 10000.0; // Square Root of Two

			struct appdata{
				float4 vertex : POSITION;
				float3 texcoord : TEXCOORD0;
			};

			struct v2f{
				float4 vertex : SV_POSITION;
				float4 screenPos : TEXCOORD0;
				float3 ray : TEXCOORD1;
			};

			struct RaycastHit {
				float3 position;
				float3 normal;
				float dist;
				int object;
			};

			// Gold Noise function
			//
			float seed;
			float gold_noise(float2 coordinate){
				float f = frac(sin(dot(coordinate*seed, float2(1.61803398874989484820459 * 00000.1, 3.14159265358979323846264 * 00000.1)))*1.41421356237309504880169 * 10000.0);
				seed = f;
				return f;
			}

			// Source
			// http://www.gamedev.net/topic/592001-random-number-generation-based-on-time-in-hlsl/
			// Supposebly from the NVidia Direct3D10 SDK
			// Slightly modified for my purposes
			struct NumberGenerator {
				int seed; // Used to generate values.

				// Returns the current random float.
				float GetCurrentFloat() {
					Cycle();
					return RANDOM_AM * seed;
				}

				// Returns the current random int.
				int GetCurrentInt() {
					Cycle();
					return seed;
				}

				// Generates the next number in the sequence.
				void Cycle() {  
					seed ^= RANDOM_MASK;
					int k = seed / RANDOM_IQ;
					seed = RANDOM_IA * (seed - k * RANDOM_IQ ) - RANDOM_IR * k;

					if (seed < 0 ) 
					seed += RANDOM_IM;

					seed ^= RANDOM_MASK;
				}

				// Cycles the generator based on the input count. Useful for generating a thread unique seed.
				// PERFORMANCE - O(N)
				void Cycle(const uint _count) {
					for (uint i = 0; i < _count; ++i)
					Cycle();
				}

				// Returns a random float within the input range.
				float GetRandomFloat(const float low, const float high) {
					float v = GetCurrentFloat();
					return low * ( 1.0f - v ) + high * v;
				}

				// Sets the seed
				void SetSeed(const uint value) {
					seed = int(value);
					Cycle();
				}
			};

			float3 xyz_to_rgb(float xc, float yc, float zc)
			{
				float xr, yr, zr, xg, yg, zg, xb, yb, zb;
				float xw, yw, zw;
				float rx, ry, rz, gx, gy, gz, bx, by, bz;
				float rw, gw, bw;

				xr = 0.67;    yr = 0.33;    zr = 1 - (xr + yr);
				xg = 0.21;    yg = 0.71;    zg = 1 - (xg + yg);
				xb = 0.14;    yb = 0.08;    zb = 1 - (xb + yb);

				xw = 0.3101;  yw = 0.3162;  zw = 1 - (xw + yw);

				/* xyz -> rgb matrix, before scaling to white. */

				rx = (yg * zb) - (yb * zg);  ry = (xb * zg) - (xg * zb);  rz = (xg * yb) - (xb * yg);
				gx = (yb * zr) - (yr * zb);  gy = (xr * zb) - (xb * zr);  gz = (xb * yr) - (xr * yb);
				bx = (yr * zg) - (yg * zr);  by = (xg * zr) - (xr * zg);  bz = (xr * yg) - (xg * yr);

				/* White scaling factors.
				Dividing by yw scales the white luminance to unity, as conventional. */

				rw = ((rx * xw) + (ry * yw) + (rz * zw)) / yw;
				gw = ((gx * xw) + (gy * yw) + (gz * zw)) / yw;
				bw = ((bx * xw) + (by * yw) + (bz * zw)) / yw;

				/* xyz -> rgb matrix, correctly scaled to white. */

				rx = rx / rw;  ry = ry / rw;  rz = rz / rw;
				gx = gx / gw;  gy = gy / gw;  gz = gz / gw;
				bx = bx / bw;  by = by / bw;  bz = bz / bw;

				/* rgb of the desired point */

				float r = (rx * xc) + (ry * yc) + (rz * zc);
				float g = (gx * xc) + (gy * yc) + (gz * zc);
				float b = (bx * xc) + (by * yc) + (bz * zc);
				return float3(r,g,b);
			}

			float3 wavelength_to_rgb(float wavelength, float gamma){
				float R, G, B, attenuation;				
			    if (wavelength >= 380 && wavelength <= 440){
			        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380);
			        R = pow((-(wavelength - 440) / (440 - 380)) * attenuation, gamma);
			        G = 0.0;
			        B = pow(1.0 * attenuation, gamma);
			    }else if (wavelength >= 440 && wavelength <= 490){
			        R = 0.0;
			        G = pow((wavelength - 440) / (490 - 440), gamma);
			        B = 1.0;
			    }else if (wavelength >= 490 && wavelength <= 510){
			        R = 0.0;
			        G = 1.0;
			        B = pow(-(wavelength - 510) / (510 - 490), gamma);
			    }else if (wavelength >= 510 && wavelength <= 580){
			        R = pow((wavelength - 510) / (580 - 510), gamma);
			        G = 1.0;
			        B = 0.0;
			    }else if (wavelength >= 580 && wavelength <= 645){
			        R = 1.0;
			        G = pow(-(wavelength - 645) / (645 - 580), gamma);
			        B = 0.0;
			    }else if (wavelength >= 645 && wavelength <= 750){
			        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645);
			        R = pow(1.0 * attenuation, gamma);
			        G = 0.0;
			        B = 0.0;
			    }else{
			        R = 0.0;
			        G = 0.0;
			        B = 0.0;
			    }
			    return float3(R,G,B);
		    }

			int _ObjectCount;
			int _VertexWidth;
			int _TriangleWidth;
			int _SampleCount;
			int _ApertureEdges;

			uint rng_state;

			float _Depth;
			float _LightCount = 1;
			float _Gamma;
			float _Aperture;
			float _FocalDepth;
			float _AutoFocus;

			float4 _Color;
			float4 _B;
			float4 _C;

			float _ObjectTypes[256];
			float _VertexCounts[256];
			float _TriangleCounts[256];
			float _Brightnesses[256];
			float _Indices[256];
			float _Lights[256];
		
			const float p = 0.5*M_PI;

			float4 _Colors[256];

			float4x4 _Transformations[256];
			float4x4 _InverseTransformations[256];

			sampler2D _Vertices;
			sampler2D _Normals;
			sampler2D _Triangles;
			sampler2D _Noise;

			samplerCUBE _Skybox;

			float rand(float2 co)
			{
				float a = 12.9898;
				float b = 78.233;
				float c = 43758.5453;
				float dt= dot(co.xy ,float2(a,b));
				float sn= fmod(dt, M_PI);
				return frac(sin(sn) * c);
			}

			float rand_xorshift()
			{
				// Xorshift algorithm from George Marsaglia's paper
				rng_state ^= (rng_state << 13);
				rng_state ^= (rng_state >> 17);
				rng_state ^= (rng_state << 5);
				return (float)rng_state / 2147483647;
			}

			float3x3 AngleAxis(float angle, float3 axis){
				float c = cos(angle);
				float s = sin(angle);
				float t = 1-c;
				float3 a = normalize(axis);
				return 
				c*float3x3(
					float3(1,0,0), 
					float3(0,1,0), 
					float3(0,0,1)) + 
				t * float3x3(
					float3(a.x*a.x, a.x*a.y, a.x*a.z),
					float3(a.y*a.x, a.y*a.y, a.y*a.z),
					float3(a.z*a.x, a.z*a.y, a.z*a.z)) +
				s * float3x3(
					float3(0, -a.z, a.y),
					float3(a.z, 0, -a.x),
					float3(-a.y, a.x, 0)
					);
			}

			float SphereIntersect(float3 origin, float3 direction) {
				float b = dot(origin, direction); 
				float c = dot(origin, origin) - 0.25; 

				// Exit if r’s origin outside s (c > 0) and r pointing away from s (b > 0) 
				if (c > 0.0 && b > 0.0) 
				return 0; 
				float discr = b*b - c; 

				// A negative discriminant corresponds to ray missing sphere 
				if (discr < 0.0) 
				return 0; 

				// Ray now found to intersect sphere, compute smallest t value of intersection
				float t = -b - sqrt(discr); 

				// If t is negative, ray started inside sphere so clamp t to zero 
				if (t < 0.0)
				t = 2*dot(origin, -direction) - t;
				return t;
			}

			float BoxIntersect(float3 origin, float3 direction) {
				float3 inv_direction = 1 / direction;
				float tx1 = (-0.5 - origin.x)*inv_direction.x;
				float tx2 = (0.5 - origin.x)*inv_direction.x;

				float tmin = min(tx1, tx2);
				float tmax = max(tx1, tx2);

				float ty1 = (-0.5 - origin.y)*inv_direction.y;
				float ty2 = (0.5 - origin.y)*inv_direction.y;

				tmin = max(tmin, min(ty1, ty2));
				tmax = min(tmax, max(ty1, ty2));

				float tz1 = (-0.5 - origin.z)*inv_direction.z;
				float tz2 = (0.5 - origin.z)*inv_direction.z;

				tmin = max(tmin, min(tz1, tz2));
				tmax = min(tmax, max(tz1, tz2));

				if (tmax >= tmin){
					if (tmin < 0){
						return tmax;
					}
					return tmin;
					}else{
						return 0;
					}
				}

				float3 TriIntersect(float3 origin, float3 direction, float3 A, float3 B, float3 C){
					float EPSILON = 0.0000001f; 
					float3 vertex0 = A;
					float3 vertex1 = B;  
					float3 vertex2 = C;
					float3 edge1, edge2, h, s, q;
					float a,f,u,v;
					edge1 = vertex1 - vertex0;
					edge2 = vertex2 - vertex0;
					h = cross(direction, edge2);
					a = dot(edge1, h);
					if (a > -EPSILON && a < EPSILON)
					return 0;
					f = 1/a;
					s = origin - vertex0;
					u = f * (dot(s, h));
					if (u < 0.0 || u > 1.0)
					return 0;
					q = cross(s, edge1);
					v = f * dot(direction, q);
					if (v < 0.0 || u + v > 1.0)
					return 0;
					// At this stage we can compute t to find out where the intersection point is on the line.
					float t = f * dot(edge2, q);
					return float3(t, u, v);
				}

				float rand_bad(float2 co){
					return frac(sin(dot(co.xy ,float2(12.9898,78.233))) * 43758.5453);
				}

				float gaussian(float2 uv){
					return sqrt( - 2.0 * log( gold_noise(uv) ) ) * cos( 2 * M_PI * gold_noise(uv) );
				/*
				float x1 = rand(seed.xy + float2(_SinTime.w - 0.73905190359, _CosTime.w + 0.591395103));
				float x2 = rand(seed.xy + float2(_SinTime.w + 0.123578901235, _CosTime.w - 0.8093851935));
				return sqrt( - 2.0 * log(x1) ) * cos( 2 * M_PI * x2 );
				*/
			}

			float3 randomOnHemisphere(float3 normal, float2 uv){
				float3 dir = float3(0,0,0);
				[loop]
				do{
					float u = gold_noise(uv);
					float v = gold_noise(uv);
					float theta = 2*M_PI*u;
					float phi = acos(2*v - 1);
					dir = float3(cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi));
					}while(abs(length(dir)-1.0) > 0.1);
					if (dot(dir, normal) < 0)
					dir = -dir;
					return dir;
				}

				float3 randomOnCosine(float3 normal, NumberGenerator random){

				/*
				float3 rand_dir = normalize(float3(gaussian(seed.xy + float2(_SinTime.w + 0.4829891, _CosTime.w + 0.90310)), gaussian(seed.xy + float2(_SinTime.w - 0.190350, _CosTime.w + 0.193010)), gaussian(seed.xy + float2(_SinTime.w - 0.109356, _CosTime.w - 0.913016))));
				float3 axis = cross(normal, rand_dir);
				float angle = asin(rand(seed + float2(_SinTime.w - 0.139103510, _CosTime.w + 0.8913578919))*2.0 - 1.0);
				//angle = rand(seed + float2(_SinTime.w - 0.139103510, _CosTime.w + 0.8913578919))*2.0 - 1.0;
				return mul(AngleAxis(angle, axis), normal);
				//return float3(1,1,1)*angle;
				*/
			}
			
			float3 randomPointInsideObject(int object, float2 seed){

				int type = _ObjectTypes[object];
				float3 pt = float3(0,0,0);
				if (type == 0){
					float theta = rand(seed.xy + float2(_SinTime.w + 0.290359235899, _CosTime.w + 0.2346998346890))*2*M_PI;
					float phi = rand(seed.xy + float2(_SinTime.w - 0.12580299, _CosTime.w + 0.23599280525))*M_PI - (M_PI/2);
					float r = rand(seed.xy + float2(_SinTime.w - 0.20352958095, _CosTime.w + 0.9063946290));
					float3 pt_prime = float3(r*cos(theta)*cos(phi), r*sin(phi), r*sin(theta)*cos(phi));
					pt = mul(_Transformations[object], float4(pt_prime, 1));
				}
				else if (type == 1){
					float x = rand(seed.xy + float2(_SinTime.w + 0.195091359, _CosTime.w + 0.13901938598)) - 0.5;
					float y = rand(seed.xy + float2(_SinTime.w + 0.890122893582, _CosTime.w + 0.346923046)) - 0.5;
					float z = rand(seed.xy + float2(_SinTime.w + 0.2171289239, _CosTime.w + 0.23891295823)) - 0.5;
					float3 pt_prime = float3(x,y,z);
					pt = mul(_Transformations[object], float4(pt_prime, 1));
				}
				return pt;
			}

			RaycastHit Raycast(float3 origin, float3 direction){
				bool first_hit = true;
				RaycastHit hit;
				hit.dist = 0;
				[loop]
				for (int object=0; object<_ObjectCount; ++object){
					float type = _ObjectTypes[object];
					if (type != 2){ //Not a mesh
						float4x4 transformation = _Transformations[object];
						float4x4 inv_transformation = _InverseTransformations[object];
						float4x4 inv_direction_transformation = float4x4(float4(inv_transformation[0].xyz, 0), 
							float4(inv_transformation[1].xyz, 0), 
							float4(inv_transformation[2].xyz, 0),
							float4(inv_transformation[3].xyz, 1));
						float3 origin_prime = mul(inv_transformation, float4(origin,1));
						float3 direction_prime = mul(inv_direction_transformation, float4(direction,1));
						direction_prime = normalize(direction_prime);
						float t_prime = 0;
						float3 hit_prime, normal_prime;
						if (type == 0){ //Sphere
							t_prime = SphereIntersect(origin_prime, direction_prime);
							hit_prime = origin_prime + t_prime*direction_prime;
							normal_prime = hit_prime;
							/*
							if (length(origin_prime) < 0.5){ //If origin was inside object
								normal_prime = -normal_prime;
							}
							*/
						}
						else if (type == 1){ //Cube
							t_prime = BoxIntersect(origin_prime, direction_prime);
							hit_prime = origin_prime + t_prime*direction_prime;
							normal_prime = hit_prime;
							for (int i=0; i<3; ++i){
								if (abs(normal_prime[i]) >= abs(normal_prime[(i+1)%3]) && abs(normal_prime[i]) >= abs(normal_prime[(i+2)%3])){
									normal_prime[i] = sign(normal_prime[i]);
								}
								else{
									normal_prime[i] = 0;
								}
							}
						}
						if (t_prime > 0){
							float3 pos = mul(transformation, float4(hit_prime, 1));
							float t = length(pos - origin);
							if (first_hit || t < hit.dist){
								first_hit = false;
								hit.position = pos;
								hit.dist = t;
								hit.normal = mul(normal_prime, inv_transformation);
								hit.normal = normalize(hit.normal);
								hit.object = object;
							}
						}
					}
				}
				return hit;
			}

			bool ptInTriangle(float3 p, float3 p0, float3 p1) {
				float dX = p.x;
				float dY = p.y;
				float dX21 = -p1.x;
				float dY12 = p1.y;
				float D = dY12*(p0.x) + dX21*(p0.y);
				float s = dY12*dX + dX21*dY;
				float t = (-p0.y)*dX + (p0.x)*dY;
				if (D<0) 
				return s<=0 && t<=0 && s+t>=D;
				return s>=0 && t>=0 && s+t<=D;
			}

			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.screenPos = ComputeScreenPos(o.vertex);
				o.ray = mul(unity_ObjectToWorld, v.vertex).xyz - _WorldSpaceCameraPos.xyz;
				return o;
			}

			fixed4 frag (v2f i) : SV_Target{

				if (_AutoFocus){
					float3 origin = _WorldSpaceCameraPos;
					float3 direction = normalize(-UNITY_MATRIX_V[2].xyz);
					RaycastHit focus = Raycast(origin, direction);
					float dist = focus.dist;
					/*
					for (int i=0; i<_Depth && focus.dist > 0 && _Colors[focus.object].a < 1; i++){
						float3 normal = focus.normal;
						float n_2 = _Indices[focus.object];
						float n_1 = 1.0;
						if (dot(direction, normal) > 0){
							//return float4(1,0,0,1);
							//n_2 = 1.0/n_2;
							n_1 = n_2;
							n_2 = 1.0;
							normal = -normal;
						}

						float ti = acos(abs(dot(direction, normal)));
						if (ti >= asin(n_2/n_1)){
							direction = direction + 2*(dot(direction, -normal)*normal);
							origin = focus.position + direction * 0.001;
							focus = Raycast(origin, direction);
						}else{
							direction = -normal + (direction - normal * dot(normal, direction)) / length((direction - normal * dot(normal, direction))) * tan(asin(n_1 / n_2 * sin(acos(length(dot(normal, direction))))));
							direction = normalize(direction);
							origin = focus.position + direction * 0.001;
							focus = Raycast(origin, direction);
						}
						dist += focus.dist;
					}
					*/
					if (focus.dist != 0){
						_FocalDepth = dist;
					}
				}

				float2 uv = i.screenPos.xy / i.screenPos.w;
				//seed = tex2Dlod(_Noise, float4(uv, 0, 0)) + _Time.w;
				//uv += float2(98219, 34623);
				float2 noise = (frac(sin(dot(uv ,float2(12.9898,78.233)*2.0)) * 43758.5453));
				abs(noise.x + noise.y) * 0.5;
				NumberGenerator rand;

				rand.seed = (tex2Dlod(_Noise, float4(uv, 0, 0)) + fmod(_Time.w, 1)/* + abs(noise.x + noise.y) * 0.5*/)/2 *2147483647;
				
				float3 camera_origin = _WorldSpaceCameraPos;
				float3 camera_direction = normalize(i.ray);
				float3 camera_forward = normalize(-UNITY_MATRIX_V[2].xyz);
				float f_dist = _FocalDepth / dot(camera_direction, camera_forward);
				float3 focal_plane_point = camera_origin + camera_direction * f_dist;
				float3 camera_up = normalize(-UNITY_MATRIX_V[1].xyz);
				float3 camera_right = cross(camera_up, camera_forward);

				if (_AutoFocus){
					RaycastHit focus = Raycast(camera_origin, camera_forward);
					if (focus.dist != 0){
						_FocalDepth = focus.dist;
					}
				}

				float3 final_color = float3(0,0,0);
				//if (uv.x < 0.5){
				[loop]
				for (int sample=0; sample<_SampleCount; ++sample){
					int k = rand.GetCurrentInt() % _ApertureEdges;
					float a = (2.0 * M_PI) / _ApertureEdges * (float)k;
					float b = (2.0 * M_PI) / _ApertureEdges * (float)(k+1);
					float3 e1 = cos(a) * camera_up + sin(a) * camera_right;
					float3 e2 = cos(b) * camera_up + sin(b) * camera_right;

					float3 p = rand.GetCurrentFloat()*e1 + rand.GetCurrentFloat()*e2;
					if (!ptInTriangle(p,e1,e2)){
						p = -p + e1+e2;
					}

					float3 origin = camera_origin + p * _FocalDepth/_Aperture;
					float3 direction = normalize(focal_plane_point - origin);
					
					RaycastHit hit = Raycast(origin, direction);
					bool dispersion = false;
					float lambda = rand.GetRandomFloat(0.38,0.78);

					for (int i=0; i<_Depth && hit.dist > 0 && _Colors[hit.object].a < 1; i++){
						float3 normal = hit.normal;
						//float n_2 = _Indices[hit.object];
						float n_1 = 1.0;
						dispersion = true;
						float lambda2 = lambda * lambda;
						float n_2 = sqrt(1.0 + (_B.x*lambda2)/(lambda2-_C.x) + (_B.y*lambda2)/(lambda2-_C.y) + (_B.z*lambda2)/(lambda2 - _C.z));


						if (dot(direction, normal) > 0){
							//return float4(1,0,0,1);
							//n_2 = 1.0/n_2;
							n_1 = n_2;
							n_2 = 1.0;
							normal = -normal;
						}



						float ti = acos(abs(dot(direction, normal)));
						if (ti >= asin(n_2/n_1)){
							direction = direction + 2*(dot(direction, -normal)*normal);
							origin = hit.position + direction * 0.001;
							hit = Raycast(origin, direction);
						}else{
							float s = sin(ti);
							float c = cos(ti);
							float f = n_1/n_2;
							float A = n_1 * c;
							float B = n_2 * sqrt(1 - pow(f*s, 2));
							float Rs = pow(abs((A-B)/(A+B)), 2);
							A = n_1 * sqrt(1 - pow(f * s, 2));
							B = n_2 * c;
							float Rp = pow(abs((A-B)/(A+B)), 2);
							float R = 0.5*(Rs+Rp);
							//float T = 1-R;
							//return float4(R,R,R,1);
							if (rand.GetCurrentFloat() <= R){
								direction = direction + 2*(dot(direction, -normal)*normal);
								origin = hit.position + direction * 0.001;
								hit = Raycast(origin, direction);
							}else{
								//direction = normal + (direction - normal * dot(normal, direction)) / abs(direction - normal * dot(normal, direction)) * tan(acos(dot(normal, direction)) / n) * sign(tan(acos(dot(normal, direction)) / n));
								//direction += hit.normal * 0.2;
								direction = -normal + (direction - normal * dot(normal, direction)) / length((direction - normal * dot(normal, direction))) * tan(asin(n_1 / n_2 * sin(acos(length(dot(normal, direction))))));
								direction = normalize(direction);
								origin = hit.position + direction * 0.001;
								hit = Raycast(origin, direction);
							}
						}
					}
					if (hit.dist == 0){
						float3 color = texCUBElod(_Skybox, float4(direction, 0)).rgb;
						if (dispersion){
							/*
							int i = ((int)(lambda*1000) - 380)/5;
							float cie_colour_match[81][3] = {{0.0014,0.0000,0.0065}, {0.0022,0.0001,0.0105}, {0.0042,0.0001,0.0201},
															{0.0076,0.0002,0.0362}, {0.0143,0.0004,0.0679}, {0.0232,0.0006,0.1102},
															{0.0435,0.0012,0.2074}, {0.0776,0.0022,0.3713}, {0.1344,0.0040,0.6456},
															{0.2148,0.0073,1.0391}, {0.2839,0.0116,1.3856}, {0.3285,0.0168,1.6230},
															{0.3483,0.0230,1.7471}, {0.3481,0.0298,1.7826}, {0.3362,0.0380,1.7721},
															{0.3187,0.0480,1.7441}, {0.2908,0.0600,1.6692}, {0.2511,0.0739,1.5281},
															{0.1954,0.0910,1.2876}, {0.1421,0.1126,1.0419}, {0.0956,0.1390,0.8130},
															{0.0580,0.1693,0.6162}, {0.0320,0.2080,0.4652}, {0.0147,0.2586,0.3533},
															{0.0049,0.3230,0.2720}, {0.0024,0.4073,0.2123}, {0.0093,0.5030,0.1582},
															{0.0291,0.6082,0.1117}, {0.0633,0.7100,0.0782}, {0.1096,0.7932,0.0573},
															{0.1655,0.8620,0.0422}, {0.2257,0.9149,0.0298}, {0.2904,0.9540,0.0203},
															{0.3597,0.9803,0.0134}, {0.4334,0.9950,0.0087}, {0.5121,1.0000,0.0057},
															{0.5945,0.9950,0.0039}, {0.6784,0.9786,0.0027}, {0.7621,0.9520,0.0021},
															{0.8425,0.9154,0.0018}, {0.9163,0.8700,0.0017}, {0.9786,0.8163,0.0014},
															{1.0263,0.7570,0.0011}, {1.0567,0.6949,0.0010}, {1.0622,0.6310,0.0008},
															{1.0456,0.5668,0.0006}, {1.0026,0.5030,0.0003}, {0.9384,0.4412,0.0002},
															{0.8544,0.3810,0.0002}, {0.7514,0.3210,0.0001}, {0.6424,0.2650,0.0000},
															{0.5419,0.2170,0.0000}, {0.4479,0.1750,0.0000}, {0.3608,0.1382,0.0000},
															{0.2835,0.1070,0.0000}, {0.2187,0.0816,0.0000}, {0.1649,0.0610,0.0000},
															{0.1212,0.0446,0.0000}, {0.0874,0.0320,0.0000}, {0.0636,0.0232,0.0000},
															{0.0468,0.0170,0.0000}, {0.0329,0.0119,0.0000}, {0.0227,0.0082,0.0000},
															{0.0158,0.0057,0.0000}, {0.0114,0.0041,0.0000}, {0.0081,0.0029,0.0000},
															{0.0058,0.0021,0.0000}, {0.0041,0.0015,0.0000}, {0.0029,0.0010,0.0000},
															{0.0020,0.0007,0.0000}, {0.0014,0.0005,0.0000}, {0.0010,0.0004,0.0000},
															{0.0007,0.0002,0.0000}, {0.0005,0.0002,0.0000}, {0.0003,0.0001,0.0000},
															{0.0002,0.0001,0.0000}, {0.0002,0.0001,0.0000}, {0.0001,0.0000,0.0000},
															{0.0001,0.0000,0.0000}, {0.0001,0.0000,0.0000}, {0.0000,0.0000,0.0000}};
							float X = cie_colour_match[i][0];
							float Y = cie_colour_match[i][1];
							float Z = cie_colour_match[i][2];
							color *= xyz_to_rgb(X, Y, Z);
							*/
							//color *= wavelength_to_rgb(lambda*1000, _Gamma);
						}
						final_color += color;
					}else{
						float3 color = float3(0,0,0);
						
						[loop]
						for (int i=0; i<_LightCount; ++i){
							int light_object = _Lights[i];
							if (light_object == hit.object){
								color += _Colors[light_object] * _Brightnesses[light_object];
							}else{
								float u = 0;
								while (u==0 || u==1)
								u = rand.GetCurrentFloat();
								float v = 0;
								while(v==0 || v==1)
								v = rand.GetCurrentFloat();
								float r = pow(rand.GetCurrentFloat()*0.125, 1.0/3);
								float theta = 2*M_PI*u;
								float phi = acos(2*v - 1);
								float3 pos = mul(_Transformations[light_object], float4(r*cos(theta)*sin(phi), r*sin(theta)*sin(phi), r*cos(phi), 1)).xyz;
								//float3 pos = mul(_Transformations[light_object], float4(0,0,0,1));
								float3 start = hit.position;
								float3 dir = normalize(pos - start);
								if (dot(dir, hit.normal) > 0){
									//return float4((dir + float3(1,1,1))/2, 1);
									start += dir * 0.001;
									RaycastHit light_hit = Raycast(start, dir);
									if (light_hit.object == light_object){
										color += _Colors[hit.object] * _Colors[light_object] * _Brightnesses[light_object] * pow(dot(hit.normal, dir), 3) / 10;
									}
								}
							}
						}
						if (dispersion){
							/*
							int i = ((int)(lambda*1000) - 380)/5;
							float cie_colour_match[81][3] = {{0.0014,0.0000,0.0065}, {0.0022,0.0001,0.0105}, {0.0042,0.0001,0.0201},
															{0.0076,0.0002,0.0362}, {0.0143,0.0004,0.0679}, {0.0232,0.0006,0.1102},
															{0.0435,0.0012,0.2074}, {0.0776,0.0022,0.3713}, {0.1344,0.0040,0.6456},
															{0.2148,0.0073,1.0391}, {0.2839,0.0116,1.3856}, {0.3285,0.0168,1.6230},
															{0.3483,0.0230,1.7471}, {0.3481,0.0298,1.7826}, {0.3362,0.0380,1.7721},
															{0.3187,0.0480,1.7441}, {0.2908,0.0600,1.6692}, {0.2511,0.0739,1.5281},
															{0.1954,0.0910,1.2876}, {0.1421,0.1126,1.0419}, {0.0956,0.1390,0.8130},
															{0.0580,0.1693,0.6162}, {0.0320,0.2080,0.4652}, {0.0147,0.2586,0.3533},
															{0.0049,0.3230,0.2720}, {0.0024,0.4073,0.2123}, {0.0093,0.5030,0.1582},
															{0.0291,0.6082,0.1117}, {0.0633,0.7100,0.0782}, {0.1096,0.7932,0.0573},
															{0.1655,0.8620,0.0422}, {0.2257,0.9149,0.0298}, {0.2904,0.9540,0.0203},
															{0.3597,0.9803,0.0134}, {0.4334,0.9950,0.0087}, {0.5121,1.0000,0.0057},
															{0.5945,0.9950,0.0039}, {0.6784,0.9786,0.0027}, {0.7621,0.9520,0.0021},
															{0.8425,0.9154,0.0018}, {0.9163,0.8700,0.0017}, {0.9786,0.8163,0.0014},
															{1.0263,0.7570,0.0011}, {1.0567,0.6949,0.0010}, {1.0622,0.6310,0.0008},
															{1.0456,0.5668,0.0006}, {1.0026,0.5030,0.0003}, {0.9384,0.4412,0.0002},
															{0.8544,0.3810,0.0002}, {0.7514,0.3210,0.0001}, {0.6424,0.2650,0.0000},
															{0.5419,0.2170,0.0000}, {0.4479,0.1750,0.0000}, {0.3608,0.1382,0.0000},
															{0.2835,0.1070,0.0000}, {0.2187,0.0816,0.0000}, {0.1649,0.0610,0.0000},
															{0.1212,0.0446,0.0000}, {0.0874,0.0320,0.0000}, {0.0636,0.0232,0.0000},
															{0.0468,0.0170,0.0000}, {0.0329,0.0119,0.0000}, {0.0227,0.0082,0.0000},
															{0.0158,0.0057,0.0000}, {0.0114,0.0041,0.0000}, {0.0081,0.0029,0.0000},
															{0.0058,0.0021,0.0000}, {0.0041,0.0015,0.0000}, {0.0029,0.0010,0.0000},
															{0.0020,0.0007,0.0000}, {0.0014,0.0005,0.0000}, {0.0010,0.0004,0.0000},
															{0.0007,0.0002,0.0000}, {0.0005,0.0002,0.0000}, {0.0003,0.0001,0.0000},
															{0.0002,0.0001,0.0000}, {0.0002,0.0001,0.0000}, {0.0001,0.0000,0.0000},
															{0.0001,0.0000,0.0000}, {0.0001,0.0000,0.0000}, {0.0000,0.0000,0.0000}};
							float X = cie_colour_match[i][0];
							float Y = cie_colour_match[i][1];
							float Z = cie_colour_match[i][2];
							color *= xyz_to_rgb(X, Y, Z);
							*/
							//color *= wavelength_to_rgb(lambda*1000, _Gamma);
						}
						final_color += color;
					}
				}
				return float4(final_color/_SampleCount, 1);
				
				/*}else{
					[loop]
					for (int sample=0; sample<_SampleCount; ++sample){
						RaycastHit hit = Raycast(origin, direction);
						float3 color = float3(0,0,0);
						if (_Brightnesses[hit.object] > 0){
							color = _Colors[hit.object] * _Brightnesses[hit.object];
						}else{
							float u = 0;
							while (u==0 || u==1)
								u = rand.GetCurrentFloat();
							float v = 0;
							while(v==0 || v==1)
								v = rand.GetCurrentFloat();
							float theta = 2*M_PI*u;
							float phi = acos(2*v - 1);
							float3 dir = float3(cos(theta)*sin(phi), sin(theta)*sin(phi), cos(phi));
							if (dot(dir, hit.normal) < 0)
								dir = -dir;
							float3 start = hit.position + dir*0.001;
							RaycastHit hit2 = Raycast(start, dir);
							if (_Brightnesses[hit2.object] > 0){
								color = (_Colors[hit.object]/M_PI) * (_Brightnesses[hit2.object] * _Colors[hit2.object]) * dot(hit.normal, dir) / (1.0/(2 * M_PI));
							}
						}
						final_color += color;
					}
					return float4(final_color/_SampleCount, 1);
					}*/
				/*
				seed = 4173.209681*_SinTime.y + 1093.1369019*_CosTime.y + _Time.w;
				float2 uv = i.screenPos.xy/i.screenPos.w;
				float3 final_colors[20];
				NumberGenerator random;
				random.seed = rand(i.screenPos.xy/i.screenPos.w + float2(_SinTime.w, _CosTime.w))*RANDOM_IM;
				rng_state = rand(i.screenPos.xy/i.screenPos.w) * 2147483647;
				[loop]
				for (int sample=0; sample<_SampleCount; ++sample){
					float3 albedos[20];
					float3 lights[20];
					float3 emits[20];
					float3 origin = _WorldSpaceCameraPos;
					float3 direction = normalize(i.ray);
					int hit_count = 0;
					float3 final_color = float3(0,0,0);
					[loop]
					for (int depth=0; depth < _Depth; ++depth){
						
						RaycastHit hit = Raycast(origin, direction);
						if (hit.dist == 0){
							//final_color = texCUBElod(_Skybox, float4(direction, 0)).rgb;
							final_color = float3(0,0,0);
							break;
						}
						//final_color = strength * min_color + (1-strength) * final_color;

						direction = randomOnHemisphere(hit.normal, uv);

						return float4(length(direction),0,0,1);
						origin = hit.position + direction * 0.0001;
						albedos[hit_count] = _Colors[hit.object] * dot(direction, hit.normal);
						emits[hit_count] = _Brightnesses[hit.object]*_Colors[hit.object].xyz;
						hit_count++;
					}
					[loop]
					for (int j=hit_count-1; j>=0; j--){
						final_color = final_color * albedos[j] + emits[j];
					}
					final_colors[sample] = final_color;
				}
				*/
				/*
				float3 final_color = float3(0,0,0);
				for (int j=0; j<_SampleCount; ++j){
					final_color += final_colors[j];
				}
				final_color /= _SampleCount;

				//return float4(final_color / (float3(1,1,1)+final_color), 1);
				return float4(final_color, 1);
				*/
				return float4(0,0,0,1);
			}
			ENDCG
		}
	}
}
