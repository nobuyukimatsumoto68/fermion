# nu=3
# for tautil2 in {0..20..1}
# do
# x=`echo "scale=3 ; 0.9+${tautil2}/200" | bc`
# echo $x
# ./eig.o $nu $x
# done

for nu in {2..4..1}
do
    echo $nu
    ./eig.o $nu
done
