all:
	./BasisSet.py -e 1 -p 24

clean:
	rm -rf *.log *.out *.gjf *.txt Gau*.* __pycache__
