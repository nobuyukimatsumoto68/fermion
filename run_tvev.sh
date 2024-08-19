make
# for mult in {12..32..4}
# do
for nu in {2..4..1}
do
    # Lx=`echo "scale=3 ; 6*$mult" | bc`
    # echo $Lx
    echo $nu
    ./solve.o $nu
    ./t_vev_v2.o $nu $Lx
done
# done
