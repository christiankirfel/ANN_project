source /etc/profile

cd /jwd
module load anaconda/2020.02-py37
tar xf ${BUDDY}/tf2.1_gpu.tar.gz
source activate /jwd/tf2.1_gpu

python PlotOnBAF.py

cd /jwd