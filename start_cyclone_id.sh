#!/bin/sh
#SBATCH -J job_py      # Specify job name
#SBATCH -p parallel    # Specify partition name
#SBATCH -N 1
#SBATCH --mem 95G
#SBATCH -c 2
#SBATCH -t 1440       # Set a limit on the total run time
#SBATCH -A m2_jgu-iceext      # Charge resources on this project account
#SBATCH --mail-user=tluettm@uni-mainz.de
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tluettm/project/logs/job_ncl.o%j    # File name for standard output
#SBATCH --error=/home/tluettm/project/logs/job_ncl.o%j

job_limit () {
    # Test for single positive integer input
    if (( $# == 1 )) && [[ $1 =~ ^[1-9][0-9]*$ ]]
    then

        # Check number of running jobs
        joblist=($(jobs -rp))
        while (( ${#joblist[*]} >= $1 ))
        do

            # Wait for any job to finis
            command='wait '${joblist[0]}
            for job in ${joblist[@]:1}
            do
                command+=' || wait '$job
            done
            eval $command
            joblist=($(jobs -rp))
        done
   fi
}

job_max_number=32

expn=$1

basedir=$HOME/cyclone_identification


module purge
module load lang/Python/3.7.4-GCCcore-8.3.0
source $basedir/basecikit/bin/activate



diri="/home/tluettm/project/"${expn}"/lagranto/"
#filename="P20140213_1200"
toponame=${diri}"ICONCONST"

plot_dir=${diri}/plots/


if [ ! -d ${plot_dir} ]; then
    mkdir ${plot_dir}
fi

minlat=45
maxlat=70
minlon=-50
maxlon=30

files=(${diri}P*)

i=0
for filename in "${files[@]:0:1}"; do

let i=$i+1
echo $filename
plotname=${plot_dir}$i.pdf

#plotname=test'.pdf'
#python ${basedir}/cyclone_id.py ${filename} --topofile ${toponame} -p ${plotname} -p_red

plotname='pres.pdf'

python ${basedir}/cyclone_id.py ${filename} --topofile ${toponame} -p ${plotname} #& \
    #   -minlat $minlat -maxlat $maxlat -minlon $minlon -maxlon $maxlon 

job_limit $job_max_number
done
wait


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


echo "Finished"
