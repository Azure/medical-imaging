import os
import logging
import json
import numpy
import torch
from monai.networks.nets import SegResNet
from monai.data import Dataset
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, ConcatItemsd, NormalizeIntensityd, Orientationd, Spacingd, Spacing, EnsureTyped, EnsureChannelFirstd
import torch
from monai.inferers import sliding_window_inference
import base64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
VAL_AMP = True

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    global model
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)

    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/extra_files/best_metric_model.pth"
    )

    state_dict = torch.load(model_path, map_location=torch.device(device))

    model.load_state_dict(state_dict)

    logging.info("Init complete")

def inference(input):

    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    
    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")
    data = json.loads(raw_data)["data"][0]

    flair_image = base64.b64decode(data['flair'])
    t1_image = base64.b64decode(data['t1'])
    t1ce_image = base64.b64decode(data['t1ce'])
    t2_image = base64.b64decode(data['t2'])

    # Write binary data to file with appropriate file extension
    with open('flair.nii.gz', 'wb') as f:
        f.write(flair_image)

    with open('t1.nii.gz', 'wb') as f:
        f.write(t1_image)

    with open('t1ce.nii.gz', 'wb') as f:
        f.write(t1ce_image)
    
    with open('t2.nii.gz', 'wb') as f:
        f.write(t2_image)

    val_transform = Compose(
    [
        LoadImaged(keys=["flair", "t1", "t1ce", "t2"]),
        EnsureChannelFirstd(keys=["flair", "t1", "t1ce", "t2"]),
        ConcatItemsd(keys=["flair", "t1", "t1ce", "t2"], name="image", dim=0),
        EnsureTyped(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    
    post_trans = Compose(
        [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )

    data_list = [{'flair': 'flair.nii.gz', 't1':'t1.nii.gz', 't1ce': 't1ce.nii.gz', 't2': 't2.nii.gz'}]
    
    val_ds = Dataset(data=data_list, transform=val_transform)

    model.eval()
    with torch.no_grad():
        val_input = val_ds[0]['image'].unsqueeze(0).to(device)
        val_output = inference(val_input)
        val_output = post_trans(val_output[0])
        result = val_output[:, :, :, :].detach().cpu().numpy()

    logging.info("Request processed")
    return result.tolist()