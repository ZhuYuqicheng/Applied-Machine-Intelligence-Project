# Website Guide

## How to run the image of our project locally:

1. Connect to VPN -> To get data from Montel API in real time
2. Make sure docker is successfully installed locally
3. Open local Terminal and run the command: docker login gitlab.ldv.ei.tum.de:5005
4. Run the command: docker run --rm -d  -p 8888:8888/tcp gitlab.ldv.ei.tum.de:5005/ami2021/group09/pred09_arima_trf:latest
    (The purpose of this step is to pull and run our container)
5. Open the browser: http://127.0.0.1:8888/ami/
    
> If all goes well, you will be able to see our page

Notes:

1. Ensure docker is installed and VPN is connected
2. Make sure Port 8888 is not occupied before running our image
3. It will take some time while executing step 4, please be patient

## How to run the image of our project on kubernetes:

1. Connect to VPN -> To get data from Montel API in real time
2. Open browser: https://coop.eikon.tum.de/
3. Add the file: group09_config.yaml
    > Path to this file: branch web_with_client -> group09/web (or: branch web_arima_trf -> group09/web)
4. Change the namespace: default -> group09
    > The page will jump to: https://coop.eikon.tum.de/#/overview?namespace=group09
5. Add two files: kubernetes-deploy.yaml and kubernetes-service.yaml
    > Path to these two files: branch web_with_client -> group09/web

6. Open browser: http://10.195.6.13:30003/ami/

> If all goes well, you will be able to see our page

Notes:

1. The difference between branch web_with_client and branch web_arima_trf is that container built from code in web_with_client is our web application with client.py and lient_p.py, while container built from web_arima_trf is only our web application.
2. The functions needed to run client.py and client_p.py are in the folder model_09
3. When the above container is running, client_p.py and client.py will also be running.  If the container does not run due to running client.py unsuccessfully, go back to step 5 and use the yaml file in branch web_arima_trf
4. If you want to run client.py and client_p.py, without running container, you can download client.py, client_p.py and folder model_09 directly from branch web_with_client -> group09/web and run them. (Folder model_09 contains the functions used by client.py and client_p.py.)

File save path: Under folder Web/ in branch master is the source code for our web application. Images are built under Packages&Registries->Container Registry, including:

- gitlab.ldv.ei.tum.de:5005/ami2021/group09/pred09_with_client:latest (image for our web with client.py)
- gitlab.ldv.ei.tum.de:5005/ami2021/group09/pred09_arima_trf:latest (image for our web without client.py)
## A few points about the web page：

The web contains some pictures and texts about the project, under the heading OUR MODELS are our models.
1. You can click on the TUM icon to go back to the home page
2. Move the mouse over the question mark in OUR MODELS to get more information.
3. Select the date, time, and prediction range, and click "predict" to start the prediction
4. Stop running the container in terminal (or delete the sevice if you run it on k8s) as you finish test this time, otherwise the web will probably show the predicted images of both models at the same time as you re-open the browser.