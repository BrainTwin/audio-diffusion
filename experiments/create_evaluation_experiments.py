import os

# Define the lists of values to iterate over
model_names = [
    "ds_64_64", "ds_256_256",
    "ssd_64_64", "ssd_256_256",
    "mc_64_64", "mc_256_256"
]

checkpoints = [
    "model_step_100", "model_step_500", "model_step_1000",
    "model_step_2500", "model_step_5000", "model_step_10000",
    "model_step_20000", "model_step_40000"
]

inference_configs = [
    "sch_ddpm_nisteps_1000", "sch_ddim_nisteps_100"
]

# Mapping model prefixes to datasets
model_to_dataset = {
    "ds": "drum_samples",
    "ssd": "spotify_sleep_dataset",
    "mc": "musiccaps"
}

# Directory to save the generated SLURM scripts
output_dir = 'experiments/base_evaluation_experiments'
os.makedirs(output_dir, exist_ok=True)

# Template for the SLURM script without comments
slurm_template = """#!/bin/bash

#SBATCH -J {job_name}
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mail-type=ALL
#SBATCH --output={output_file}
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
conda activate audiodiff_env

application="accelerate"

options="launch --config_file /home/th716/rds/hpc-work/audio-diffusion/config/accelerate_local.yaml \
/home/th716/rds/hpc-work/audio-diffusion/scripts/evaluation.py \
--reference_paths {reference_paths} \
--generated_path /home/th716/rds/hpc-work/audio-diffusion/models/{model_name}/{model_step_X}/samples/audio/{sample_path} \
--log_dir /home/th716/rds/hpc-work/audio-diffusion/models/{model_name}/{model_step_X}/samples/{sample_path} \
--metric frechet_audio_distance \
--model_names clap-laion-audio clap-laion-music vggish clap-2023 encodec-emb-48k encodec-emb MERT-v1-95M"

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

# Iterate over the combinations and create SLURM scripts
for model_name in model_names:
    for checkpoint in checkpoints:
        for sample_path in inference_configs:
            # Determine the dataset based on the model prefix
            prefix = model_name.split('_')[0]
            dataset_name = model_to_dataset.get(prefix)
            
            job_name = f"base_experiment_{model_name}_{checkpoint}_{sample_path}"
            output_file = f"experiments/hpc_runs/{job_name}.out"
            
            # Generate the reference paths, ensuring no duplicates
            reference_paths = [
                f"/home/th716/rds/hpc-work/audio-diffusion/cache/{dataset_name}/waveform",
                "/home/th716/rds/hpc-work/audio-diffusion/cache/fma_pop/waveform",
                "/home/th716/rds/hpc-work/audio-diffusion/cache/musiccaps/waveform"
            ]
            if dataset_name == "musiccaps":
                reference_paths = list(set(reference_paths))  # Remove duplicates
            reference_paths_str = " ".join(reference_paths)

            # Generate the script content
            script_content = slurm_template.format(
                job_name=job_name,
                output_file=output_file,
                reference_paths=reference_paths_str,
                model_name=model_name,
                model_step_X=checkpoint,
                sample_path=sample_path
            )
            
            # Save the script to a file
            script_filename = os.path.join(output_dir, f"{job_name}.wilkes3")
            with open(script_filename, 'w') as script_file:
                script_file.write(script_content)
            
            print(f"Generated {script_filename}")
