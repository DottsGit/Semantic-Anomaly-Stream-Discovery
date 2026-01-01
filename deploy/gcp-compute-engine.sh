# GCP Compute Engine deployment with GPU
# This script creates a GPU-enabled VM and deploys SOSD

# Variables - customize these
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
ZONE="us-central1-a"
INSTANCE_NAME="sosd-gpu-instance"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1

# Create instance with GPU
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --maintenance-policy=TERMINATE \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform \
    --tags=sosd,http-server

# Wait for instance to be ready
echo "Waiting for instance to be ready..."
sleep 60

# SSH into instance and setup
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    # Clone repository
    git clone https://github.com/yourrepo/semantic-object-stream-discovery.git
    cd semantic-object-stream-discovery

    # Install dependencies
    pip install -e .

    # Run pipeline
    python -m src.main --help
"

echo "Instance created: $INSTANCE_NAME"
echo "SSH: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
