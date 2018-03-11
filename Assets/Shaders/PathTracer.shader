Shader "Custom/PathTracer"
{
	Properties{
		_Color ("Color", Color) = (0,0,0,1)
		_Brightness ("Brightness", Float) = 0
		_Skybox ("Skybox", CUBE) = "white" {}
		_Depth ("Depth", Range (1, 20)) = 5
		_SampleCount ("Sample Count", Range (1, 50)) = 5
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

			int _ObjectCount;
			int _VertexWidth;
			int _TriangleWidth;
			int _SampleCount;

			uint rng_state;

			float _Depth;
			float _LightCount = 1;

			float4 _Color;

			float _ObjectTypes[256];
			float _VertexCounts[256];
			float _TriangleCounts[256];
			float _Brightnesses[256];
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

				if (tmax >= tmin)
					return tmin;
				else{
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
							if (length(origin_prime) < 0.5){
								normal_prime = -normal_prime;
							}
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

			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.screenPos = ComputeScreenPos(o.vertex);
				o.ray = mul(unity_ObjectToWorld, v.vertex).xyz - _WorldSpaceCameraPos.xyz;
				return o;
			}

			fixed4 frag (v2f i) : SV_Target{
				float2 uv = i.screenPos.xy / i.screenPos.w;
				//seed = tex2Dlod(_Noise, float4(uv, 0, 0)) + _Time.w;
				//uv += float2(98219, 34623);
				NumberGenerator rand;
				rand.seed = (tex2Dlod(_Noise, float4(uv, 0, 0)) + fmod(_Time.w, 1))/2 *2147483647;
				
				float3 origin = _WorldSpaceCameraPos;
				float3 direction = normalize(i.ray);
				float3 final_color = float3(0,0,0);
				if (uv.x < 0.5){
					[loop]
					for (int sample=0; sample<_SampleCount; ++sample){
						RaycastHit hit = Raycast(origin, direction);
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
								//return float4((dir + float3(1,1,1))/2, 1);
								start += dir * 0.001;
								RaycastHit light_hit = Raycast(start, dir);
								if (light_hit.object == light_object){
									color += _Colors[hit.object] * _Colors[light_object] * _Brightnesses[light_object] * pow(dot(hit.normal, dir), 3) / 10;
								}
							}
						}
						final_color += color;
					}
					return float4(final_color/_SampleCount, 1);
				}else{
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
				}
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
