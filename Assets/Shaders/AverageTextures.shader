// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Custom/AddTextures"
{
Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_Overlay ("Overlay", 2D) = "white" {}
		_Depth ("Depth", Float) = 0
	}
	SubShader
	{
		Tags { "Queue"="Transparent" "RenderType"="Transparent" }
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			// make fog work
			
			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
			};

			struct v2f
			{
				float4 vertex : SV_POSITION;
				float4 screenPos : TEXCOORD1;
			};

			sampler2D _MainTex;
			sampler2D _Overlay;

			float _Depth;

			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.screenPos = ComputeScreenPos(o.vertex);
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				float2 uv = i.screenPos.xy / i.screenPos.w;
				float4 col1 = tex2D(_MainTex, uv);				
				float4 col2 = tex2D(_Overlay, uv);
				float4 scaled1 = (-col1)/(col1 - float4(1,1,1,1));
				float4 scaled2 = (-col2)/(col2 - float4(1,1,1,1));
				//float4 scaled_result = scaled1*_Depth/(_Depth+1) + scaled2/(_Depth+1);
				return col1+col2;
			}
			ENDCG
		}
	}
}