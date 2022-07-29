for i in 10 20 30 40 
do
    for j in 100 200 300 400 
    do
        python3 main.py -r $i -t $j
        echo "res = $i, time = $j" >> test.txt
        python3 pade.py >> test.txt
        echo "" >> test.txt
    done
done