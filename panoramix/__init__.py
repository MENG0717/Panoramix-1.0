# https://timothybramlett.com/How_to_create_a_Python_Package_with___init__py.html

# Making sure stuff works when we `import panoramix`

from .settings import *
from .data import *
from .mk_random import *
from .model import *
from .visualize import *

__author__ = "Niko Heeren"
__copyright__ = "Copyright 2018"
__credits__ = ["Niko Heeren", "Rupert Myers"]
__license__ = "TBA"
__version__ = "0.0"
__maintainer__ = "Niko Heeren"
__email__ = "niko.heeren@gmail.com"
__status__ = "raw"