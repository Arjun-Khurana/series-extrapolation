for j in 75 100 150 200 250 300 350 400 450 500
do
    mpirun -np 4 python3 main.py -t $j
    python3 pade.py -t $j
done

