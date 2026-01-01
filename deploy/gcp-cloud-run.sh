# GCP Cloud Run deployment (CPU-only, serverless)
# Good for intermittent workloads or when GPU isn't needed

# Variables
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION="us-central1"
SERVICE_NAME="sosd-pipeline"
IMAGE_NAME="gcr.io/$PROJECT_ID/sosd-pipeline:latest"

# Enable required APIs
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    pubsub.googleapis.com \
    bigquery.googleapis.com \
    --project=$PROJECT_ID

# Build and push container
echo "Building container image..."
gcloud builds submit \
    --tag $IMAGE_NAME \
    --project=$PROJECT_ID

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image=$IMAGE_NAME \
    --platform=managed \
    --region=$REGION \
    --memory=4Gi \
    --cpu=2 \
    --timeout=3600 \
    --concurrency=1 \
    --max-instances=10 \
    --set-env-vars="SOSD_FEATURES__DEVICE=cpu" \
    --set-env-vars="SOSD_OUTPUT__ENABLE_PUBSUB=true" \
    --set-env-vars="SOSD_OUTPUT__PUBSUB_PROJECT=$PROJECT_ID" \
    --allow-unauthenticated \
    --project=$PROJECT_ID

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --platform=managed \
    --region=$REGION \
    --format='value(status.url)' \
    --project=$PROJECT_ID)

echo "Deployed to: $SERVICE_URL"

# Create Pub/Sub topic for events
gcloud pubsub topics create sosd-events --project=$PROJECT_ID || true

# Create BigQuery dataset
bq mk --dataset $PROJECT_ID:sosd || true

echo "Deployment complete!"
echo "Service URL: $SERVICE_URL"
echo "Pub/Sub Topic: sosd-events"
echo "BigQuery Dataset: $PROJECT_ID:sosd"
