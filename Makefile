all:something

something:
	./BasisSet.py -e 1 -j 24

clean:
	rm -f *.log *.out *.gjf *.txt
	rm -rf __pycache__
