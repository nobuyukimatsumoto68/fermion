set xlabel 'x'
set ylabel 'y'

set title 'L=100'

# splot './t0t0.dat'

set terminal pdf
set output "plot3d.pdf"

# set contour
set zrange [1.0e-14:10.0]
set view 20,30
# set isosamples 20, 20

# set hidden3d
# set dgrid3d 100,100 qnorm 1
# set pm3d interpolate 0,0
# set dgrid3d 200,200

# set logscale x
# set logscale y
set logscale z

splot './t0t0.dat' with lines
# splot './t0t0.dat' with impulses

# show contour

