for nu in 2 3 4
do
    ./solve.o $nu
    ./xixi.o $nu
    ./eps.o $nu
    ./eps_vev.o $nu
    ./tt_v2_all.o $nu
    ./t_vev.o $nu
    ./t_vev_v2.o $nu
    ./t_epseps.o $nu
    # ./eig.o $nu
done
