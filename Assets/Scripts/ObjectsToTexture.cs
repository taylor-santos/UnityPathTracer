using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectsToTexture : MonoBehaviour {
	public Texture2D vertices;
	public Texture2D normals;
	public Texture2D triangles;
	public Texture2D noise;
	public List<MeshFilter> MFs;
	public List<Material> mats;
	public Material mat;
	public RenderTexture finalRender;
	public RenderTexture render;
	public RenderTexture RT;
	public RenderTexture RenderRT;
	private Material renderMaterial;
	private Material copyMaterial;
	private Material scaleMaterial;
	private int depth;
	private bool frame = true;
	// Use this for initialization
	void Start () {
		MFs = new List<MeshFilter>(GameObject.FindObjectsOfType<MeshFilter>() as MeshFilter[]);
		mats = new List<Material>();
		List<float> types = new List<float>();
		List<int> mesh_MFs = new List<int>();
		for (int i=0; i<MFs.Count; ++i){
			Material material = MFs[i].gameObject.GetComponent<Renderer>().material;
			mats.Add(material);
			switch(MFs[i].sharedMesh.name){
				case "Sphere":
					types.Add(0);
					break;
				case "Cube":
					types.Add(1);
					break;
				default:
					types.Add(2);
					mesh_MFs.Add(i);
					break;
			}
		}
		Shader.SetGlobalFloatArray("_ObjectTypes", types);
		List<List<Color>> vert_colors = new List<List<Color>>();
		List<List<Color>> norm_colors = new List<List<Color>>();
		List<List<Vector3>> tri_colors = new List<List<Vector3>>();
		List<float> vert_counts = new List<float>(512);
		List<float> tri_counts = new List<float>(512);
		int max_vert_count = 0;
		int max_tri_count = 0;
		for (int i=0; i<mesh_MFs.Count; ++i){
			MeshFilter MF = MFs[mesh_MFs[i]];
			vert_colors.Add(new List<Color>());
			norm_colors.Add(new List<Color>());
			tri_colors.Add(new List<Vector3>());

			vert_counts.Add(MF.sharedMesh.vertices.Length);
			tri_counts.Add(MF.sharedMesh.triangles.Length/3);
			max_vert_count = Mathf.Max(max_vert_count, MF.sharedMesh.vertices.Length);
			max_tri_count = Mathf.Max(max_tri_count, MF.sharedMesh.triangles.Length/3);
			foreach (Vector3 vert in MF.sharedMesh.vertices){
				vert_colors[i].Add(new Color(vert.x, vert.y, vert.z, 1));
			}
			foreach (Vector3 norm in MF.sharedMesh.normals){
				norm_colors[i].Add(new Color(norm.x, norm.y, norm.z, 1));
			}
			for (int j=0; j<MF.sharedMesh.triangles.Length/3; ++j){
				tri_colors[i].Add(new Vector3(MF.sharedMesh.triangles[3*j+0],
											  MF.sharedMesh.triangles[3*j+1],
											  MF.sharedMesh.triangles[3*j+2]));
			}
		}

		Color[] finalVertColors = new Color[mesh_MFs.Count*max_vert_count];
		for (int i=0; i<vert_colors.Count; ++i){
			for (int j=0; j<vert_colors[i].Count; ++j){
				finalVertColors[max_vert_count*i + j] = vert_colors[i][j];
			}
		}

		Color[] finalNormColors = new Color[mesh_MFs.Count*max_vert_count];
		for (int i=0; i<norm_colors.Count; ++i){
			for (int j=0; j<norm_colors[i].Count; ++j){
				finalNormColors[max_vert_count*i + j] = norm_colors[i][j];
			}
		}

		Color[] finalTriColors = new Color[mesh_MFs.Count*max_tri_count];
		for (int i=0; i<tri_colors.Count; ++i){
			for (int j=0; j<tri_colors[i].Count; ++j){
				Vector3 tri = tri_colors[i][j];
				finalTriColors[max_tri_count*i + j] = new Color((tri.x+0.5f) / max_vert_count, (tri.y+0.5f) / max_vert_count, (tri.z+0.5f) / max_vert_count);
			}
		}

		vertices = new Texture2D(max_vert_count, mesh_MFs.Count, TextureFormat.RGBAFloat, false);
		vertices.filterMode = FilterMode.Point;
		vertices.SetPixels(finalVertColors);
		vertices.Apply();
		//Shader.SetGlobalTexture("_Vertices", vertices);
		Shader.SetGlobalTexture("_Vertices", vertices);

		normals = new Texture2D(max_vert_count, mesh_MFs.Count, TextureFormat.RGBAFloat, false);
		normals.filterMode = FilterMode.Point;
		normals.SetPixels(finalNormColors);
		normals.Apply();
		//Shader.SetGlobalTexture("_Normals", normals);
		Shader.SetGlobalTexture("_Normals", normals);

		triangles = new Texture2D(max_tri_count, mesh_MFs.Count, TextureFormat.RGBAFloat, false);
		triangles.filterMode = FilterMode.Point;
		triangles.SetPixels(finalTriColors);
		triangles.Apply();
		//Shader.SetGlobalTexture("_Triangles", triangles);
		Shader.SetGlobalTexture("_Triangles", triangles);

		noise = new Texture2D(Screen.width, Screen.height, TextureFormat.RGBAFloat, false);
		noise.filterMode = FilterMode.Point;
		Color[] noise_pixels = new Color[Screen.width*Screen.height];
		for (int i=0; i<Screen.width*Screen.height; ++i){
			noise_pixels[i] = new Color(1,1,1,0)*Random.value + new Color(0,0,0,1);
		}
		noise.SetPixels(noise_pixels);
		noise.Apply();
		Shader.SetGlobalTexture("_Noise", noise);
		/*
		Shader.SetGlobalInt("_ObjectCount", MFs.Count);
		Shader.SetGlobalInt("_VertexWidth", max_vert_count);
		Shader.SetGlobalInt("_TriangleWidth", max_tri_count);
		Shader.SetGlobalFloatArray("_VertexCounts", vert_counts.ToArray());
		Shader.SetGlobalFloatArray("_TriangleCounts", new float[2]{12,13});
		*/
		Shader.SetGlobalInt("_ObjectCount", MFs.Count);
		Shader.SetGlobalInt("_VertexWidth", max_vert_count);
		Shader.SetGlobalInt("_TriangleWidth", max_tri_count);
		Shader.SetGlobalFloatArray("_VertexCounts", vert_counts.ToArray());
		Shader.SetGlobalFloatArray("_TriangleCounts", tri_counts.ToArray());		

		RT = new RenderTexture(Screen.width, Screen.height, 32);
		RT.format = RenderTextureFormat.ARGBFloat;
		RT.name = name + " RenderTexture";
		RT.antiAliasing = 1;
		RT.filterMode = FilterMode.Point;

		Camera cam = GetComponent<Camera>();
		cam.targetTexture = RT;
		
		render = new RenderTexture(Screen.width, Screen.height, 32);
		render.format = RenderTextureFormat.ARGBFloat;
		render.name = "Render";
		render.antiAliasing = 1;
		render.filterMode = FilterMode.Point;

		finalRender = new RenderTexture(Screen.width, Screen.height, 32);
		finalRender.format = RenderTextureFormat.ARGBFloat;
		finalRender.name = "Final Render";
		finalRender.antiAliasing = 1;
		finalRender.filterMode = FilterMode.Point;

		RenderRT = new RenderTexture(Screen.width, Screen.height, 32);
		RenderRT.format = RenderTextureFormat.ARGBFloat;
		RenderRT.name = name + " RenderTexture";
		RenderRT.antiAliasing = 1;
		RenderRT.filterMode = FilterMode.Point;

		renderMaterial = new Material(Shader.Find("Custom/AddTextures"));
		copyMaterial = new Material(Shader.Find("Custom/CopyTexture"));
		scaleMaterial = new Material(Shader.Find("Custom/ScalePixels"));

		depth = 0;
	}

	// Update is called once per frame
	void Update () {
		List<Matrix4x4> transformations = new List<Matrix4x4>();
		List<Matrix4x4> inverse_transformations = new List<Matrix4x4>();
		List<Vector4> colors = new List<Vector4>();
		List<float> indices = new List<float>();
		List<float> brightnesses = new List<float>();
		List<float> lights = new List<float>();
		for (int i=0; i<MFs.Count; ++i){
			MeshFilter MF = MFs[i];
			transformations.Add(MF.transform.localToWorldMatrix);
			inverse_transformations.Add(MF.transform.localToWorldMatrix.inverse);
			colors.Add((Vector4)mats[i].color);
			indices.Add(mats[i].GetFloat("_Index"));
			float brightness = mats[i].GetFloat("_Brightness");
			brightnesses.Add(Mathf.Max(0, brightness));
			if (brightness > 0){
				lights.Add(i);
			}
		}
		Shader.SetGlobalMatrixArray("_Transformations", transformations);
		Shader.SetGlobalMatrixArray("_InverseTransformations", inverse_transformations);
		Shader.SetGlobalVectorArray("_Colors", colors);
		Shader.SetGlobalFloatArray("_Indices", indices);
		Shader.SetGlobalFloatArray("_Brightnesses", brightnesses);
		Shader.SetGlobalFloatArray("_Lights", lights);
		Shader.SetGlobalFloat("_LightCount", lights.Count);
		frame = true;
	}
	
	void OnPostRender(){
		if (frame){
			depth++;
			frame = false;
			renderMaterial.SetTexture("_Overlay", RT);
			renderMaterial.SetFloat("_Depth", depth);
			Graphics.Blit(render, RenderRT, renderMaterial);
			Graphics.Blit(RenderRT, render, copyMaterial);
			scaleMaterial.SetFloat("_Depth", depth);
			Graphics.Blit(render, finalRender, scaleMaterial);
			
		}
		
	}
	
	void OnGUI() {
		Graphics.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), finalRender);
	}
}
