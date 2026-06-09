# How to reproduce the project

## Table of Contents
- [1. Google Cloud Platform and Terraform](#1-google-cloud-platform-and-terraform)
   - [1.1. Google Cloud Platform (GCP)](#11-google-cloud-platform-gcp)
   - [1.2. Terraform](#12-terraform)
- [2. Mage And Data ETL](#2-mage-and-data-etl)
- [3. Analytics Engineering](#3-analytics-engineering)
- [4. Spark ETL](#4-spark)
- [5. Kafka](#5-kafka)


## 1. Google Cloud Platform and Terraform

### 1.1. Google Cloud Platform (GCP)

__GCP Initial Setup__

- 1.Create an account with your Google email ID
- 2.Setup your first project if you haven't already
- 3.Setup service account & authentication for this project
    - Grant Viewer role to begin with.
    - Download service-account-keys (.json) for authentication.
- 4.Download SDK for local setup
- 5.Set environment variable to point to your downloaded GCP keys

__Setup for Access__
 
- 1.[IAM Roles](https://cloud.google.com/storage/docs/access-control/iam-roles) for Service account:
   * Go to the *IAM* section of *IAM & Admin* https://console.cloud.google.com/iam-admin/iam
   * Click the *Edit principal* icon for your service account.
   * Add these roles in addition to *Viewer* : **Storage Admin** + **Storage Object Admin** + **BigQuery Admin**  + **Service Account User**  + **Cloud Run Admin** + **Artifact Registry Administrator** + **Artifact Registry Repository Administrator**

   _NOTE: You might need to add more roles depending on your use case._

   
- 2.Enable these APIs for your project:
   * https://console.cloud.google.com/apis/library/iam.googleapis.com
   * https://console.cloud.google.com/apis/library/iamcredentials.googleapis.com

   _NOTE: You might need to enable more APIs depending on your use case._
   
- 3.Please ensure `GOOGLE_APPLICATION_CREDENTIALS` env-var is set.
   ```shell
   export GOOGLE_APPLICATION_CREDENTIALS="<path/to/your/service-account-authkeys>.json"
   ```

__Google Cloud SDK Authentication__

Set `GOOGLE_APPLICATION_CREDENTIALS` to point to the file
```bash
export GOOGLE_APPLICATION_CREDENTIALS={your_path}/{your_file}.json
```

Now authenticate your SDK with your GCP account
```bash
gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
```


### 1.2. Terraform

Go to the `terraform` folder and run the following commands:
```bash
# Initialize state file (.tfstate)
terraform init

# Check changes to new infra plan
terraform plan -var="project=<your-gcp-project-id>"

# Or just
terraform plan

# Apply changes to new infra
terraform apply

# Delete infra after your work, to avoid costs on any running services
terraform destroy
```


## 2. ETL
### Download service account key json file:

Go to Google cloud platform -> IAM & Admin -> Service account.
Choose or create service account with the following roles:
- Storage Object Viewer
- Storage Object Creator
- Storage Object Admin
- BigQuery Data Viewer
- BigQuery Data Editor
- BigQuery Job User

Go to Keys and download service account key json file

### Deploy Mage to Cloud Run

See: [link](https://docs.mage.ai/production/deploying-to-cloud/gcp/setup)

### Create and schedule ETL pipeline:

#### Copy service account key json file to Mage file system (see ETL/code/Dockerfile)

#### Set up Climate Data Store API
```bash
cd
touch .cdsapirc
nano .cdsapirc
```

#### Add key
```bash
url: https://cds-beta.climate.copernicus.eu/api
key: <<your_key>>
```

#### Create 2 pipeline:

Create weather pipeline (All the measurement except for rain) and rain pipeline (see .py files in ETL/code) to extract data from API and load to GCS.


Go to Pipeline -> Choose a pipeline -> Trigger to schedule crawl data daily.

## 3. Analytics Engineering

- See the [SQL](https://github.com/BIG-DATA-IT4931/analyze_weather_data/tree/main/big-query) file for the SQL queries.
- Go to [this link](http://192.168.1.12:8501) to see the dashboard.


## 4. Spark
```bash
cd ./spark/spark-3.3.2-bin-hadoop3
```
Creating new master node:
```bash
./sbin/start-master.sh
```
Check if the Spark master is running by visiting the Web UI: http://<external-ip>:8080.

Creating new worker node:
```bash
./sbin/start-worker.sh spark://<external-ip>:7077
```
To confirm that the workers are running:
```bash
jps
```
Stop master node: 
```bash
./sbin/stop-master.sh
```

Stop worker node:
```bash
./sbin/stop-worker.sh
```


## 5. Kafka

Run the following command to setup Kafka and Zookeeper:
```bash
# Connect to VM
gcloud compute ssh kafka-instance

#Set up Climate Data Store API
cd
touch .cdsapirc
nano .cdsapirc

#Add key
url: https://cds-beta.climate.copernicus.eu/api
key: <<your_key>>

#submit key 
Ctrl S + Ctil X 

# Install OpenJDK 11 (required for Spark)
wget https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz
tar xzfv openjdk-11.0.2_linux-x64_bin.tar.gz
export JAVA_HOME="${HOME}/spark/jdk-11.0.2"
export PATH="${JAVA_HOME}/bin:${PATH}"


rm openjdk-11.0.2_linux-x64_bin.tar.gz  # Clean up archive

# Install Spark (version 3.3.2)
wget https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz
tar xzfv spark-3.3.2-bin-hadoop3.tgz
export SPARK_HOME="${HOME}/spark/spark-3.3.2-bin-hadoop3"
export PATH="${SPARK_HOME}/bin:${PATH}"
rm spark-3.3.2-bin-hadoop3.tgz  # Clean up archive

# Install Google Cloud Storage connector for Hadoop
wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-2.2.5.jar

# Install BigQuery connector for Spark
wget https://repo1.maven.org/maven2/com/google/cloud/spark/spark-bigquery-with-dependencies_2.12/0.24.0/spark-bigquery-with-dependencies_2.12-0.24.0.jar

# Clone the repo
git clone https://github.com/BIG-DATA-IT4931/analyze_weather_data.git
cd analyze_weather_data
bash ./setup/vm_setup.sh
exec newgrp docker
# Set the external IP of the VM, you must base it on the actual address on your virtual machine
export KAFKA_ADDRESS=34.124.241.190

# Start kafka
cd kafka
pip install -r requirements.txt
docker-compose build
docker-compose up -d

# Turn off kafka
docker-compose down

# Start producer
python src/producers/producer_run.py 



export GCP_GCS_BUCKET=dtc_data_lake_rock-data-436815-e2


# Run consumer
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.3 --jars $SPARK_HOME/jars/spark-bigquery-with-dependencies_2.12-0.24.0.jar,$SPARK_HOME/jars/gcs-connector-hadoop3-2.2.5.jar src/consumers/consumer.py
```
