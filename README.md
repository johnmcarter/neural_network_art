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
for more explanation about the input parameter options. There is a default value of 1 for style weight, 1e-5 for content weight, and the program saves the output to images/output.png by default. 

To reproduce the images in Figure 1 in the paper, run:
```
python3 neural_network_art.py images/bridge.jpg images/stonehenge.jpg --sw 1 --cw 0.00000001
python3 neural_network_art.py images/photographer.jpg images/stonehenge.jpg --sw 1 --cw 0.00000001
python3 neural_network_art.py images/cannes.jpg images/stonehenge.jpg --sw 1 --cw 0.00000001
```
To reproduce the images in Figure 2, run:
```
python3 neural_network_art.py images/crucifixion.jpg images/sydney.jpg --sw 1 --cw <CONTENT WEIGHTS>
python3 neural_network_art.py images/montsv.jpg images/monserrat.jpg --sw <STYLE WEIGHTS> --cw 1
```
Replace CONTENT WEIGHTS with one of the following values as desired: 0.001, 0.00001, .0000000001, .000000000000001
Replace STYLE WEIGHTS with one of the following values as desired: 10, 1000, 100000, 10000000, 1000000000