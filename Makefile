default: homework1.py
	python homework1.py

check:
	python --version | xargs -n1 | tail -n1 | grep '^3' || echo "use python 3"
	python -c 'import scipy, matplotlib'

