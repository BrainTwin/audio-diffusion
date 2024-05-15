import os

# Define the base directory to save the scripts
base_dir = 'experiments/base_inference_experiments'

# Ensure the directory exists
os.makedirs(base_dir, exist_ok=True)

# List of model base paths
model_configs = [
    "ds_64_64", "ds_256_256",
    "ssd_64_64", "ssd_256_256",
    "mc_64_64", "mc_256_256"
]

base_path = "/home/th716/rds/hpc-work/audio-diffusion/models/{config}/model_step_x"

# List of model checkpoints
checkpoints = [
    "model_step_100", "model_step_500", "model_step_1000",
    "model_step_2500", "model_step_5000", "model_step_10000",
    "model_step_20000", "model_step_40000"
]

# Inference configurations
inference_configs = [
    ("--num_inference_steps 1000 --scheduler ddpm", "1000_ddpm"),
    ("--num_inference_steps 100 --scheduler ddim", "100_ddim")
]

# Template for the slurm script
slurm_template = """#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mail-type=ALL
#SBATCH --output={output_path}
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load miniconda/3

source ~/.bashrc
conda init bash
conda activate audiodiff_env

application="accelerate"
options="launch --config_file /home/th716/rds/hpc-work/audio-diffusion/config/accelerate_local.yaml /home/th716/rds/hpc-work/audio-diffusion/scripts/inference_unet.py --pretrained_model_path {model_path} --num_images 1024 --eval_batch_size 64 {inference_options}"

CMD="$application $options"
workdir="/home/th716/rds/hpc-work/audio-diffusion"

cd $workdir
echo "Changed directory to `pwd`."
echo "JobID: $SLURM_JOB_ID"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
    export NODEFILE=`generate_pbs_nodefile`
    cat $NODEFILE | uniq > machine.file.$SLURM_JOB_ID
    echo "Nodes allocated:"
    echo `cat machine.file.$SLURM_JOB_ID | sed -e 's/\..*$//g'`
fi

echo "Executing command:"
echo "$CMD"
eval $CMD
"""

# Generate each script
for config in model_configs:
    model_path_template = base_path.format(config=config)
    for checkpoint in checkpoints:
        full_model_path = model_path_template.replace('model_step_x', checkpoint)
        for inf_config, suffix in inference_configs:
            # Include model config in the job name and output file path
            job_name = f"{config}_{checkpoint}_{suffix}"
            script_name = f"{job_name}.wilkes3"
            output_path = os.path.join(base_dir, f"{job_name}.out")
            
            # Format the slurm script with specific configuration
            script_content = slurm_template.format(
                job_name=job_name,
                output_path=output_path,
                model_path=full_model_path,
                inference_options=inf_config
            )
            
            # Write the script to a file
            with open(os.path.join(base_dir, script_name), 'w') as script_file:
                script_file.write(script_content)

            print(f"Generated script: {script_name}")
