using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Refraction : MonoBehaviour {

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		RaycastHit hit;
		if (Physics.Raycast(transform.position, transform.forward, out hit, Mathf.Infinity)){
			Debug.DrawLine(transform.position, hit.point, Color.white);
			Debug.DrawRay(hit.point, hit.normal, Color.cyan);
			Vector3 direction = transform.forward;
			Vector3 normal = hit.normal;
			float n_1 = 1f;
			float n_2 = 1.5f;
			direction = -normal + (direction - normal * Vector3.Dot(normal, direction)) / ((direction - normal * Vector3.Dot(normal, direction))).magnitude * Mathf.Tan(Mathf.Asin(n_1 / n_2 * Mathf.Sin(Mathf.Acos(Mathf.Abs(Vector3.Dot(normal, direction))))));
			Debug.DrawRay(hit.point, direction*100, Color.green);
		}
	}
}
