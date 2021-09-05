# Code generated by `stubgen`. DO NOT EDIT.
from kubernetes import client as client, config as config
from kubernetes.client.api_client import ApiClient as ApiClient
from kubernetes.client.rest import ApiException as ApiException
from typing import Any

class ConfigMapLock:
    api_instance: Any
    leader_electionrecord_annotationkey: str
    name: Any
    namespace: Any
    identity: Any
    configmap_reference: Any
    lock_record: Any
    def __init__(self, name, namespace, identity) -> None: ...
    def get(self, name, namespace): ...
    def create(self, name, namespace, election_record): ...
    def update(self, name, namespace, updated_record): ...
    def get_lock_object(self, lock_record): ...
    def get_lock_dict(self, leader_election_record): ...