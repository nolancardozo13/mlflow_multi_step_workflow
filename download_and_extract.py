import os
import mlflow
import argparse

def download_and_extract(kaggle_url, folder_name):
    with mlflow.start_run() as mlrun:
        os.system('cp kaggle.json /home/nolan/.kaggle/kaggle.json')
        os.system('chmod 600 /home/nolan/.kaggle/kaggle.json')
        os.system(kaggle_url)
        os.system('mkdir '+ folder_name)
        os.system('unzip aptos2019-blindness-detection.zip -d '+ folder_name)
        os.system('rm aptos2019-blindness-detection.zip')
        mlflow.log_artifacts(folder_name,"raw_dataset_dir")
        os.system('rm -rf '+ folder_name)

if __name__ in "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle_url", default="kaggle competitions download -c aptos2019-blindness-detection", type = str, help= "The kaggle competiions dataset download syntax (kaggle cli format)")
    parser.add_argument("--folder_name", default="dataset", type = str, help= "The folder in which the downloaded datasets contants are stored")
    args = parser.parse_args()
    download_and_extract(args.kaggle_url, args.folder_name)