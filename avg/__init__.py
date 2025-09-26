
from .ewa_offline import EWAOffline
from .lawa_offline import LAWAOffline

AVG_REGISTRY = {
  "EWA": EWAOffline,
  "LAWA": LAWAOffline,
}
