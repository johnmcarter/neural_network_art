## Neural Network Art - Spring 2020  
#### CS 615 Deep Learning 
#### Jocelyn Rego & John Carter  


To run this implementation, download or clone this repository. To install the required dependencies, run 
```
pip3 install -r requirements.txt
```
in the same directory on the command line. To then run the code, run
```
python3 neural_network_art.py <style image> <content image> <style weight (optional)> <content weight (optional)> <output (optional)>
```
in the local directory on the command line. The style and content image, as well as the output, should be a path of where to find the input images and where to put the output, respectively. The images should be of type .jpg. The user can also run 
```
python3 neural_network_art.py -h
```
for more explanation about the input parameter options. There is a default value of 1 for style weight, 1e-5 for content weight, and the program saves the output to images/output.png by default. To create an example output.png file with the provided crucifixion.jpg style image and plaza_de_espana.jpg content image, run
```
python3 neural_network_art.py crucifixion.jpg plaza_de_espana.jpg
```