#!/usr/bin/env python
from train import train_
from GeometryOptimization import geo_opt
from MolecularDynamics import md
from StaticCompress import static_compress


geo_opt()
md()
static_compress()
train_()

