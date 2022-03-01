# In tamsa2, the local path of root interrupts path of root in cmssw
# so remove the local path before importing ROOT is essential (might cause problem in some stage)
import sys

def setup():
	sys.path.remove('/opt/ohpc/pub/apps/root_6_12_06/lib')
