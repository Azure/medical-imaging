# NVFlare on Azure

The Federated Learning paradigm gained a lot of interest in the healthcare community, because it enables model training on all available data, without sharing data between insitutions. 
NVIDIA Flare is a domain-agnostic, open-source and extensible SDK for federated learning. This walkthrough descibes how NVIDIA Flare can be run on Azure.

The way this demo is implemented, is that a federated server is hosted on an Azure VM. 
Three Azure Machine Learning Workspaces in different regions are used to simulate different clients (e.g. hospital sites) that train a local model on their private dataset.
The local models will be aggregated on the federated server, using federated averaging. This means that initial weights are distributed to the clients, who perform local training.
After local training, clients return their local weights to the federated server, which then aggregates these weights (averaged). This new set of global average weights is redistributed to clients and the process repeats for the specified number of rounds.

An overview of the architecture can be found below:

![Solution overview!](fedlearn.jpg "Solution overview")

## 1. Prepare Federated Server
In Federated Learning, a federated server is the central party that communicates with different clients to send training instructions and gathers information on the locally trained models.
The federated server is hosted on a Data Science Virtual Machine (DSVM) in Azure. During this step, we create a DSVM, and configure the networking to make it accessible for the clients. The way federated learning works in NVFlare, is that there is only inbound communication from the client to the host. This means we have to open the ports on the VM that the clients will send messages to.
In the default communication, we need to open ports 8002 and 8003.

- Start by opening the bash client of your choice, or use the Azure Cloud Shell: https://shell.azure.com/

- Log in to the Azure CLI using the following command: `az login`
- Select the Azure subscription of your choice to deploy the resources to by using the following command: `az account set --subscription <subscription-id>`
- Create a resource group for the different resources: `az group create -n fedlearn`
- Create a DSVM using the instruction on this page: https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro
- In order to make the DSVM accessible to the clients, we need to create a public DNS entry. This can be done as follows:
    - Select your VM in the portal.
    - In the left menu, select Properties
    - Under Public IP address\DNS name label, click on your IP address.
    - Under DNS name label, enter the prefix you want to use.
    - Select Save at the top of the page.

- Use the following command to open port 8002 - 8003 on the VM (In the setup below, the ports will be open from all outside sources. A more secure option would be open the ports only for traffic from the client IP addresses):
```
az vm open-port -g fedlearn -n fedserver --port 8002-8003 --priority 100
```
- Decide on an approach for accessing the VM. This can be done through Bastion or SSH, optionally with Just in Time (JIT) access configured

## 2. Setup federated server
In this step, we will login to the VM and setup the federated server.

- Connect to the VM via SSH by using the following command in the Windows Terminal (bash) `ssh <vm-username>@<vm-public-ip>` and fill in the VM password.
- Clone this repo to the VM and navigate to it.
- Install NVFlare library via pip: `pip install nvflare`
- Add the DNS name that is configured for the virtual machine to `project.yml`
- Run provision script:
```
provision -p project.yml
```
- Create two new folders on VM in home dir: fedserver and fedadmin
- Copy server and admin packages to these folders
- Run start script in fedserver folder
- Run the following command to open the host file or your VM: `sudo vi /etc/hosts`
- Press *i* to start insering content in your host file
- Create a new entry on line 2 using this pattern: `<private-ip-of-vm> <dns-address>` (the private IP address of your VM can be found by accessing your VM in the Azure portal and in the Overview, look for Networking -> Private IP address)


## 3. Prepare clients
As mentioned, the clients will be implemented through different Azure Machine Learning Workspaces in different regions. In this step, we are creating the workspaces and create Compute Instances within it. 
The Compute Instances will be configured as a client and initiate communication with the federated server.

For every client, repeat the following steps. Make sure to use a different region for the different Workspaces to simulate a global setup:
- Create a ML workspace: `az ml workspace create -w <workspace-name> -g <resource-group-name> --location <location>` 
- Access the workspace via the Azure Machine Learning Studio on https://ml.azure.com.
- Create Compute Instance by going to Compute -> Compute Instances and click on *new*.
- Install NVFlare library via pip: `pip install nvflare`
- We need the prepare the dataset. Open a new notebook in Azure ML, create a new cell and paste the following code:
```python
%pip install kaggle --upgrade split-folders

import matplotlib.pyplot as plt
import json
from azureml.core import Workspace, Dataset, Experiment
import splitfolders

workspace = Workspace.from_config()

# Export Kaggle configuration variables
%env KAGGLE_USERNAME=[Kaggle user name]
%env KAGGLE_KEY=[API token]

# remove folders and zipfile from previous runs of the cell
!rm /tmp/chest-xray-pneumonia.zip
!rm -r /tmp/chest_xray
!rm -r /tmp/chest_xray_tvt

# Download the Pneumonia dataset
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p /tmp

!unzip -q /tmp/chest-xray-pneumonia.zip -d /tmp

# temp download locations before registering as AML dataset
download_root = '/tmp/chest_xray/train' 
train_val_test_root = '/tmp/chest_xray_tvt/'

train_val_test_split = (0.8, 0.1, 0.1)
random_seed = 33
splitfolders.ratio(download_root, train_val_test_root, random_seed, ratio=train_val_test_split)

# check dataset splits
for split in os.listdir(train_val_test_root):
    for label in ['NORMAL', 'PNEUMONIA']:
        files = os.listdir(os.path.join(train_val_test_root, split, label))
        print(f'{split}-{label}: ', len(files))

from azureml.core import Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath

# Upload data to AzureML Datastore
ds = workspace.get_default_datastore()

ds = Dataset.File.upload_directory(src_dir=train_val_test_root,
            target=DataPath(ds, 'chest-xray'),
            show_progress=False, overwrite=False)

# Register file dataset with AzureML
ds = ds.register(workspace=workspace, name="pneumonia", description="Pneumonia train / val / test folders with 2 classes", create_new_version=True)

print(f'Dataset {ds.name} registered.')
```
- Update the Kaggle credentials in your code cell and run it. Using this code, a dataset will be downloaded from Kaggle, which will be split in a train, validation and testset and registered as a Dataset in Azure ML.
- Copy one of the site packages that is generated on the federated server to the filesystem connected to your Machine Learning Workspace. One approach for this is to go to Notebooks in the Azure Machine Learning Studio, click on the *plus*-icon, and select *Upload folder*.
- From the Machine Learning Studio, open a new Terminal (also on the Notebooks page) connected the Compute Instance you created. Navigate to the startup folder of the client package that you uploaded, and run the start script using the following command: `bash ./start.sh`.

## 4. Run the experiment
In this step, we use the admin client of our federated application to upload the training code to the different clients and start the training run.

- Open a new SSH connection to your federated server.
- Log into the Admin client on federated server, by navigating to the admin folder (where you copied the admin package) and running the start script.
- The admin client will prompt for a username and password. By default, both are *admin@nvidia.com*.
- Use the admin commands one by one to run the experiment:

```
set_run_number 1
upload_app pneumonia-federated
deploy_app pneumonia-federated
start_app all
```
- For more information on the different command, type *help* or *?* to the admin client, or read more on [this](https://nvidia.github.io/NVFlare/user_guide/admin_commands.html) page.

## 5. Access training metrics via TensorBoard
On the federated clients, training metrics are streamed to the server that can be accessed from Tensorboard. The federated server also stores metrics from the global model as TensorBoard events.

- To run TensorBoard on the federated server, use the following command: `tensorboard --logdir=workspace/server/run_1/tb_events`
- By default, TensorBoard is running on port 6006. To make this port accessible, open port 6006 on the VM for your client IP specifically.
- In your browser, nagivate to: `https://<dns-label-of-your-vm>:6006`.

## 6. Clean up resources

- Delete your resource group by using the following command: `az group delete -g <resource-group-name>`.