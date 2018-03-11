using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RayTransformTest : MonoBehaviour {
	public Transform target;
	public Matrix4x4 transformation;
	public Matrix4x4 inv_transformation;
	// Use this for initialization
	void Start () {
		
	}

	float IntersectRaySphere(Vector3 p, Vector3 d) {
		float b = Vector3.Dot(p, d); 
		float c = Vector3.Dot(p, p) - 0.25f; 

		// Exit if r’s origin outside s (c > 0) and r pointing away from s (b > 0) 
		if (c > 0.0f && b > 0.0f) 
			return 0; 
		float discr = b*b - c; 

		// A negative discriminant corresponds to ray missing sphere 
		if (discr < 0.0f) 
			return 0; 

		// Ray now found to intersect sphere, compute smallest t value of intersection
		float t = -b - Mathf.Sqrt(discr); 

		// If t is negative, ray started inside sphere so clamp t to zero 
		if (t < 0.0f){
			t = 2*Vector3.Dot(p, -d) - t;
		}
		return t;
	}
	
	// Update is called once per frame
	void Update () {
		transformation = target.localToWorldMatrix;
		inv_transformation = target.localToWorldMatrix.inverse;

		Matrix4x4 inv_direction_transformation = inv_transformation;
		inv_direction_transformation.SetColumn(3, new Vector4(0,0,0,1));

		Vector3 origin = transform.position;
		Vector3 direction = transform.forward;

		Vector3 origin_prime = inv_transformation*((Vector4)origin+new Vector4(0,0,0,1));
		Vector3 direction_prime = ((Vector3)(inv_direction_transformation*((Vector4)direction + new Vector4(0,0,0,1)))).normalized;

		float t = IntersectRaySphere(origin_prime, direction_prime);
		Vector3 hit_prime = origin_prime + t*direction_prime;
		Vector3 normal_prime = origin_prime + t*direction_prime;

		Vector3 hit = transformation * ((Vector4)hit_prime + new Vector4(0,0,0,1));

		Matrix4x4 inv_transpose_transformation = inv_transformation.transpose;
		Vector3 normal = ((Vector3)(inv_transpose_transformation * normal_prime)).normalized;

		Debug.DrawRay(transform.position, transform.forward*100, Color.black);
		Debug.DrawRay(hit, Vector3.up, Color.red);
		Debug.DrawRay(hit, normal, Color.cyan);

		Debug.DrawRay(origin_prime, direction_prime*100, Color.blue);
		Debug.DrawRay(hit_prime, normal_prime, Color.cyan);

		RaycastHit hitobj;
		if (Physics.Raycast(origin, direction*100, out hitobj)){
			Debug.DrawRay(hitobj.point, hitobj.normal, Color.blue);
		}
	}
}
