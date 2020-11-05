"""
XT init module.
We register all the system module in here.
"""
from xt.train import main as train
from xt.evaluate import main as evaluate
from xt.benchmarking import main as benchmarking
from xt.framework.register import import_all_modules_for_register
import_all_modules_for_register()

__version__ = '0.2.1'

__ALL__ = ["train", "evaluate", "benchmarking"]
