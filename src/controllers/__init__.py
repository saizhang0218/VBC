REGISTRY = {}

from .basic_controller_corridor_vbc import BasicMAC_corridor
from .basic_controller_6h_vs_8z_vbc import BasicMAC_6h_vs_8z

REGISTRY["basic_mac_corridor"] = BasicMAC_corridor
REGISTRY["basic_mac_6h_vs_8z"] = BasicMAC_6h_vs_8z