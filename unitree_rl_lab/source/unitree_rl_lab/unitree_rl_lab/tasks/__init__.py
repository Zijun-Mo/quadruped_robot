##
# Register Gym environments.
##


"""Package initializer for the unitree_rl_lab.unitree_rl_lab.tasks namespace."""

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = []
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
