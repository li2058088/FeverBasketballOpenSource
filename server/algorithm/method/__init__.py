from algorithm.method import dqn
from algorithm.method import qmix
from algorithm.method import hyq
from algorithm.method import exq
from algorithm.method import vdn
from algorithm.method import dqmix
from algorithm.method import wqmix
from algorithm.method import qplex
from algorithm.method import qtran

methods = dict(
	dqn=dqn.Method,
	qmix=qmix.Method,
	hyq=hyq.Method,
	exq=exq.Method,
	vdn=vdn.Method,
	dqmix=dqmix.Method,
	wqmix=wqmix.Method,
	qplex=qplex.Method,
	qtran=qtran.Method
)