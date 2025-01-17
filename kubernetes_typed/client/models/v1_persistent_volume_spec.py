# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1PersistentVolumeSpecType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import Dict, List

from .v1_aws_elastic_block_store_volume_source import V1AWSElasticBlockStoreVolumeSourceType
from .v1_azure_disk_volume_source import V1AzureDiskVolumeSourceType
from .v1_azure_file_persistent_volume_source import V1AzureFilePersistentVolumeSourceType
from .v1_ceph_fs_persistent_volume_source import V1CephFSPersistentVolumeSourceType
from .v1_cinder_persistent_volume_source import V1CinderPersistentVolumeSourceType
from .v1_object_reference import V1ObjectReferenceType
from .v1_csi_persistent_volume_source import V1CSIPersistentVolumeSourceType
from .v1_fc_volume_source import V1FCVolumeSourceType
from .v1_flex_persistent_volume_source import V1FlexPersistentVolumeSourceType
from .v1_flocker_volume_source import V1FlockerVolumeSourceType
from .v1_gce_persistent_disk_volume_source import V1GCEPersistentDiskVolumeSourceType
from .v1_glusterfs_persistent_volume_source import V1GlusterfsPersistentVolumeSourceType
from .v1_host_path_volume_source import V1HostPathVolumeSourceType
from .v1_iscsi_persistent_volume_source import V1ISCSIPersistentVolumeSourceType
from .v1_local_volume_source import V1LocalVolumeSourceType
from .v1_nfs_volume_source import V1NFSVolumeSourceType
from .v1_volume_node_affinity import V1VolumeNodeAffinityType
from .v1_photon_persistent_disk_volume_source import V1PhotonPersistentDiskVolumeSourceType
from .v1_portworx_volume_source import V1PortworxVolumeSourceType
from .v1_quobyte_volume_source import V1QuobyteVolumeSourceType
from .v1_rbd_persistent_volume_source import V1RBDPersistentVolumeSourceType
from .v1_scale_io_persistent_volume_source import V1ScaleIOPersistentVolumeSourceType
from .v1_storage_os_persistent_volume_source import V1StorageOSPersistentVolumeSourceType
from .v1_vsphere_virtual_disk_volume_source import V1VsphereVirtualDiskVolumeSourceType


@dataclass
class V1PersistentVolumeSpecType:
    access_modes: List[str]
    aws_elastic_block_store: V1AWSElasticBlockStoreVolumeSourceType
    azure_disk: V1AzureDiskVolumeSourceType
    azure_file: V1AzureFilePersistentVolumeSourceType
    capacity: Dict[str, str]
    cephfs: V1CephFSPersistentVolumeSourceType
    cinder: V1CinderPersistentVolumeSourceType
    claim_ref: V1ObjectReferenceType
    csi: V1CSIPersistentVolumeSourceType
    fc: V1FCVolumeSourceType
    flex_volume: V1FlexPersistentVolumeSourceType
    flocker: V1FlockerVolumeSourceType
    gce_persistent_disk: V1GCEPersistentDiskVolumeSourceType
    glusterfs: V1GlusterfsPersistentVolumeSourceType
    host_path: V1HostPathVolumeSourceType
    iscsi: V1ISCSIPersistentVolumeSourceType
    local: V1LocalVolumeSourceType
    mount_options: List[str]
    nfs: V1NFSVolumeSourceType
    node_affinity: V1VolumeNodeAffinityType
    persistent_volume_reclaim_policy: str
    photon_persistent_disk: V1PhotonPersistentDiskVolumeSourceType
    portworx_volume: V1PortworxVolumeSourceType
    quobyte: V1QuobyteVolumeSourceType
    rbd: V1RBDPersistentVolumeSourceType
    scale_io: V1ScaleIOPersistentVolumeSourceType
    storage_class_name: str
    storageos: V1StorageOSPersistentVolumeSourceType
    volume_attributes_class_name: str
    volume_mode: str
    vsphere_volume: V1VsphereVirtualDiskVolumeSourceType
