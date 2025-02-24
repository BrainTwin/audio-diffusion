import os

# Define parameter values
hop_lengths = [64, 128, 256, 512, 1024]
nffts = [256, 512, 1024, 2048, 4096]

# Base directory for SLURM scripts
slurm_dir = "/home/th716/audio-diffusion/experiments/final_experiments/evaluation/"
os.makedirs(slurm_dir, exist_ok=True)

# Template for SLURM script
slurm_template = """#!/bin/bash

#SBATCH -J evaluation_512_128_hl_{hl}_nfft_{nfft}.wilkes3
#SBATCH -A NALLAPERUMA-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=experiments/final_hpc_runs/evaluation_512_128_hl_{hl}_nfft_{nfft}.out
#SBATCH -p ampere

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load miniconda/3

source ~/.bashrc  
conda init bash
conda activate fadtk_new_env

application="~/.conda/envs/fadtk_new_env/bin/python"

options="/home/th716/rds/hpc-work/audio-diffusion/scripts/evaluation.py \
--reference_paths /home/th716/rds/hpc-work/audio-diffusion/cache/spotify_sleep_dataset/waveform \
/home/th716/rds/hpc-work/audio-diffusion/cache/fma_pop/waveform \
--generated_path /home/th716/rds/hpc-work/audio-diffusion/cache/spotify_sleep_dataset/waveform_1024/mel_spec_512_128_hl_{hl}_nfft_{nfft}/converted_gl32 \
--log_dir /home/th716/rds/hpc-work/audio-diffusion/cache/spotify_sleep_dataset/waveform_1024/mel_spec_512_128_hl_{hl}_nfft_{nfft}/converted_gl32/logs \
--metric frechet_audio_distance \
--model_names clap-laion-audio clap-laion-music vggish"

CMD="$application $options"

workdir="/home/th716/rds/hpc-work/audio-diffusion"

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
    export NODEFILE=`generate_pbs_nodefile`
    cat $NODEFILE | uniq > machine.file.$JOBID
    echo -e "\nNodes allocated:\n================"
    echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e '\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)'

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
"""

# Generate SLURM scripts
for hl in hop_lengths:
    for nfft in nffts:
        script_name = f"evaluation_hl_{hl}_nfft_{nfft}.wilkes3"
        script_path = os.path.join(slurm_dir, script_name)
        
        with open(script_path, "w") as f:
            f.write(slurm_template.format(hl=hl, nfft=nfft))
        
        print(f"Generated: {script_path}")
