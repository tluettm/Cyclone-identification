#!/bin/sh

module purge
module load anaconda3/bleeding_edge
module load ncl/6.5.0-gccsys


basedir=$HOME/cyclone_identification
plot_dir=${basedir}"/plots/"

if [ ! -d ${plot_dir} ]; then
    mkdir ${plot_dir}
fi


diri="/work/bb0994/experiments/wcb_13feb2014_phil/lagranto_data/"
#filename="P20140213_1200"
toponame=${diri}"ICONCONST"

minlat=35
maxlat=48
minlon=-40
maxlon=-20

files=(${diri}P*)

i=0
for filename in "${files[@]}"; do

let i=$i+1
echo $filename
plotname=${plot_dir}$i.pdf

python ${basedir}/cyclone_id.py ${filename} --topofile ${toponame} -p ${plotname} \
   # -minlat $minlat -maxlat $maxlat -minlon $minlon -maxlon $maxlon  #-pd
done

#convert ${plot_dir}* cy_id_merge.pdf

# minlat=35.
# maxlat=48.
# minlon=-40.
# maxlon=-20.

# minlat=45.
# maxlat=46.
# minlon=-40.
# maxlon=-39.

# minlat=30.
# maxlat=75.
# minlon=-50.
# maxlon=10.