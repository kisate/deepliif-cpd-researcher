# example: https://github.com/IBM/watson-openscale-samples/blob/main/Cloud%20Pak%20for%20Data/WML/notebooks/custom_metrics/Custom%20Metrics%20Provider%20for%20Cloud%20Pak%20for%20Data.ipynb

parms = {
        "url": 'https://cpd-cpd.apps.cpd.mskcc.org', # CANNOT end with "/", or the metrics logging api will fail with "The requested resource could not be found."
        "username": '********',
        "apikey": '********'
    }

import json
import requests
import base64
from requests.auth import HTTPBasicAuth
import time
import uuid    

headers = {}
headers["Content-Type"] = "application/json"
headers["Accept"] = "application/json"


# Get the access token
def get_access_token():
    url = '{}/icp4d-api/v1/authorize'.format(parms['url'])
    payload = {
        'username': parms['username'],
        'api_key': parms['apikey']
    }
    response = requests.post(url, headers=headers, json=payload, verify=False)
    json_data = response.json()
    access_token = json_data['token']
    return access_token


#Update the run status to Finished in the custom monitor instance
def update_monitor_instance(base_url, access_token, custom_monitor_instance_id, payload):
    monitor_instance_url = base_url + '/v2/monitor_instances/' + custom_monitor_instance_id + '?update_metadata_only=true'

    patch_payload  = [
        {
            "op": "replace",
            "path": "/parameters",
            "value": payload
        }
    ]
    headers["Authorization"] = "Bearer {}".format(access_token)
    response = requests.patch(monitor_instance_url, headers=headers, json = patch_payload, verify=False)
    monitor_response = response.json()
    return response.status_code, monitor_response

#Add your code to compute the custom metrics. 
def get_metrics(access_token, data_mart_id, subscription_id):
    #Add the logic here to compute the metrics. 
    metrics = {"specificity": 1.2, "sensitivity": 0.85}
    return metrics


# Publishes the Custom Metrics to OpenScale
def publish_metrics(base_url, access_token, data_mart_id, subscription_id, custom_monitor_id, custom_monitor_instance_id, custom_monitoring_run_id, timestamp):
    # Generate an monitoring run id, where the publishing happens against this run id
    custom_metrics = get_metrics(access_token, data_mart_id, subscription_id)
    measurements_payload = [
              {
                "timestamp": timestamp,
                "run_id": custom_monitoring_run_id,
                "metrics": [custom_metrics]
              }
            ]
    headers["Authorization"] = "Bearer {}".format(access_token)
    measurements_url = base_url + '/v2/monitor_instances/' + custom_monitor_instance_id + '/measurements'
    response = requests.post(measurements_url, headers=headers, json = measurements_payload, verify=False)
    published_measurement = response.json()

    return response.status_code, published_measurement


def score( input_data ):
    import datetime
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    payload = input_data.get("input_data")[0].get("values")
    data_mart_id = payload['data_mart_id']
    subscription_id = payload['subscription_id']
    custom_monitor_id = payload['custom_monitor_id']
    custom_monitor_instance_id = payload['custom_monitor_instance_id']
    custom_monitor_instance_params  = payload['custom_monitor_instance_params']

    base_url = parms['url'] + '/openscale' + '/' + data_mart_id
    access_token = get_access_token()

    published_measurements = []
    error_msg = []

    custom_monitoring_run_id = custom_monitor_instance_params["run_details"]["run_id"]
    try:
        status_code, published_measurement = publish_metrics(base_url, access_token, data_mart_id, subscription_id, custom_monitor_id, custom_monitor_instance_id, custom_monitoring_run_id, timestamp)
        if int(status_code) in [200, 201, 202]:
            custom_monitor_instance_params["run_details"]["run_status"] = "finished"
            published_measurements.append(published_measurement)
        else:
            custom_monitor_instance_params["run_details"]["run_status"] = "error"
            custom_monitor_instance_params["run_details"]["run_error_msg"] = published_measurement
            error_msg.append(published_measurement)
            
        custom_monitor_instance_params["last_run_time"] = timestamp
        status_code, response = update_monitor_instance(base_url, access_token, custom_monitor_instance_id, custom_monitor_instance_params)
        if not int(status_code) in [200, 201, 202]:
            error_msg.append(response)

    except Exception as ex:
        error_msg.append(str(ex))
        
    if len(error_msg) == 0:
        response_payload = {
            "predictions" : [{ 
                "values" : published_measurements
            }]

        }
    else:
        response_payload = {
            "predictions": [],
            "error_msg": error_msg,
        }

    return response_payload
