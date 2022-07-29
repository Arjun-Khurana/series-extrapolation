clean: 
	rm -f *.npz
	rm -f *.h5
	rm -f *.png
	rm -f *.txt

run: clean
	python3 main.py
	python3 pade.py

pade:
	python3 pade.py