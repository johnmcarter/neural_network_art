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
in the local directory on the command line. There is a default value of 1e5 for style weight, 1 for content weight, and the program saves the output to images/output.png by default. To create the output.png file with the provided crucifixion.jpg style image and plaza_de_espana.jpg content image, run
```
python3 neural_network_art.py crucifixion.jpg plaza_de_espana.jpg
```