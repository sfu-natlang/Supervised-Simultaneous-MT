import importlib
import os

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        model_name = file[:file.find('.py')]
        importlib.import_module('examples.Supervised_simul_MT.models.' + model_name)
