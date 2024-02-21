#1;95;0c!/bin/bash -l

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# to submit: qsub script.sh

#------ qsub options --------#
#$ -P qfe
#$ -M mtsmtnbyk@gmail.com
##### run time limit. format: hh:mm:ss; default 12 hrs
#$ -l h_rt=0:30:00
##### merge error and output
#$ -j y
##### email options; begins (b), ends (e), is aborted (a), is suspended (s), or never (n) - default
#$ -m beas

#$ -pe omp 4

# --------- job info -----------#

echo "start"
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "NSlots : $NSLOTS"
echo "Host name : $HOSTNAME"
echo "working directory : $TMPDIR"
echo "=========================================================="


#------- Program execution -------#

OMP_NUM_THREADS=$NSLOTS
echo "running program"
date
pwd
time ./psipsi.o
echo "finished"
date
