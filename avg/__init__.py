
from .ewa_offline import EWAOffline
from .lawa_offline import LAWAOffline
from .lewa_offline import LEWAOffline
from .log_spaced_lawa import LogSpacedLAWA

AVG_REGISTRY = {
  "EWA": EWAOffline,
  "LAWA": LAWAOffline,
  "LEWA": LEWAOffline,
  "LogSpacedLAWA": LogSpacedLAWA,
}
