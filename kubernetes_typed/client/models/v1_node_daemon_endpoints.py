# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1NodeDaemonEndpointsType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_daemon_endpoint import V1DaemonEndpointType


@dataclass
class V1NodeDaemonEndpointsType:
    kubelet_endpoint: V1DaemonEndpointType
