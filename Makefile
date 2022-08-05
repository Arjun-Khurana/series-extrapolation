clean: 
	rm -f *.npz
	rm -f *.h5
	rm -f *.png
	rm -f *.txt
	rm -f *.gif

run: clean
	python3 main.py
	python3 pade.py

pade:
	python3 pade.py

gif:
	python3 main.py 
	h5topng -t 0:332 -R -Zc dkbluered -a yarg -A main-eps-000000.00.h5 main-ez.h5
	convert main-ez.t*.png main-ez.gif
	mkdir gif
	mv *.png gif
	mv *.h5 gif
	mv *.gif gif

test:
	python3 test.py