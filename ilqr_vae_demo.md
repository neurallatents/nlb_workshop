The iLQR-VAE demo notebook lives inside a Docker container. This means that you have to run it inside a closed virtual environment: while it might take a little bit more than a minute to download the container, the advantage is that you won't have to deal with installing any packages once you have set up the container because everything will be ready to be used.

To access the container, you will first need to install Docker from [here](https://docs.docker.com/engine/install/). 

Once Docker is installed, you will need to pull the image containing the notebook + environment (similarly to pulling a Github repository). 
You can access the image by running <code>docker pull ghennequin/ilqr-vae:v2.0</code>.

Once you have pulled that, you can find the number of the image (and other containers you may have installed by running <code>docker images</code>.
Then, you can enter the container by running <code>docker run -it -p 8888:8888 image_name</code>. Note that if you get an error due to permissions with the '/var/run/docker.sock' file, this can fixed with a 'chmod' command.

You’ll then be inside the docker container, in the repository of the tutorial : first run <code>git pull</code> and then <code>jupyter notebook --port=8888 --no-browser --ip=0.0.0.0</code>, and you can access the notebook using the link that will be printed!

Finally, Docker images can be heavy, so if you want to remove the container after having run the tutorial you can use <code>git pulldocker image rm image_name</code>.