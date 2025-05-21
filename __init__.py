"""
This module initializes the isd package and exposes its main components.
"""

__version__ = '0.1.0'
__author__ = 'Javier Morenosa'
__email__ = 'javiersanchezmoreno2@gmail.com'


from isd.core.models import LowFrequencySeries, ISDAlgorithm


__all__ = ['LowFrequencySeries', 'ISDAlgorithm']


_CITATION = """
@article{quinn2025iterative,
  title={An Iterative Shifting Disaggregation Algorithm for Multi-Source, Irregularly Sampled, and Overlapped Time Series},
  author={Quinn, Colin O. and Brown, Ronald H. and Corliss, George F. and Povinelli, Richard J.},
  journal={Sensors},
  year={2025},
  publisher={MDPI},
  doi={10.3390/s25030895}
}
"""

def cite():
    """
    Muestra la información para citar el paquete.
    """
    print(_CITATION)
