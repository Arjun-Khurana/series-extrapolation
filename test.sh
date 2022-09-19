# for j in 50 100 150 200 250 300
# do
#     mpirun -np 4 python3 cavity.py -N 7 -t $j
#     python3 pade.py -t $j
# done

# for j in 500 1000 2000 4000 10000
# do
#     mpirun -np 4 python3 cavity.py -N 7 -t $j
# done

# python3 test.py

# for j in 10 20 30 40 50 100 200 300
# do
#     mpirun -np 4 python3 ldos.py -t $j
#     python3 pade.py -t $j
# done

# for j in 500 1000 2000 4000
# do
#     mpirun -np 4 python3 ldos.py -t $j
# done

# python3 test.py

pt=400
dt=400
gt=1000
dpt=10
ddt=500
dgt=100

f="ring-data"

# for r in 10 20 
# do
#     mkdir "${f}_res=$r"
#     mpirun -np 4 python3 ring.py -r $r -pt $pt -dt $dt -gt $gt -dpt $dpt -dpt $dpt -dgt $dgt | grep harminv >> harminv.txt
#     for ((i = $dpt; i <= $pt; i+= $dpt))
#     do
#         python3 pade.py -t $i -s 0
#     done
#     python3 comparison.py -pt $pt -dt $dt -gt $gt -f "ring-data"
#     mv *.png "ring-data_res=$r"
#     mv *.txt "ring-data_res=$r"
#     mv *.npz "ring-data_res=$r"
#     python3 sandbox.py -f $f -ff "${f}_res=$r"
# done

# for r in 10 40
# do
#     mkdir "${f}_res=$r" 
#     mpirun -np 4 python3 rods.py -r $r -pt $pt -dt $dt -gt $gt -dpt $dpt -ddt $ddt -dgt $dgt -nx 5 | grep harminv >> harminv.txt
#     for ((i = $dpt; i <= $pt; i+= $dpt))
#     do
#         python3 pade.py -t $i -s 3
#     done
#     python3 comparison.py -pt $pt -dt $dt -gt $gt -f "rods-data"
#     mv *.png "${f}_res=$r"
#     mv *.txt "${f}_res=$r"
#     mv *.npz "${f}_res=$r"
#     python3 sandbox.py -f $f -ff $"${f}_res=$r"
# done

# for r in 20
# do
#     mkdir "${f}_res=$r" 
#     mpirun -np 4 python3 ldos.py -r $r -pt $pt -dt $dt -gt $gt -dpt $dpt -ddt $ddt -dgt $dgt | grep harminv >> harminv.txt
#     for ((i = $dpt; i <= $pt; i+= $dpt))
#     do
#         python3 pade.py -t $i -s 2
#     done
#     python3 comparison.py -pt $pt -dt $dt -gt $gt -f $f
#     mv *.png "${f}_res=$r"
#     mv *.txt "${f}_res=$r"
#     mv *.npz "${f}_res=$r"
#     python3 sandbox.py -f $f -ff $"${f}_res=$r"
# done

# for s in 5
# do
#     mkdir "${f}_step=$s"
#     for ((i = $dpt; i <= $pt; i+= $dpt))
#     do
#         python3 pade.py -t $i -st $s -s 0
#     done
#     mv pade*.npz "${f}_step=$s"
#     mv *.png "${f}_step=$s"
#     cp ring*.npz "${f}_step=$s"
#     python3 sandbox.py -f $f -ff "${f}_step=$s"
# done

cd pade_julia
for ((i = 10; i <= 400; i+= 10))
do
    julia --project=. src/pade_julia.jl $i
done