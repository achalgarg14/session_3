# Assignments

## Part-1 - Creating a simple end to end Neural Network on Excel

### Basic Neural Network
A 3 layer Fully Connected Neural Network (consisting 1 Input Layer, 1 Hidden Layer and 1 Ouput Layer) is trained on MS Excel (Actually Google sheet but it is very similar to MS Excel if you haven't heard about it)

Neural network has 2 input nodes, 2 nodes in hiddlen layer as well as 2 output layers. The basic diagram of the Neural Network is as shown below:

<img width="696" alt="NN" src="https://user-images.githubusercontent.com/118976187/212166786-e4629c86-fc27-4f1a-b8fd-50f1fe2e69f6.png">



Input Layer is having 2 nodes:
1. i1
2. i2

Hidden Layer values:
1. h1 = w1\*i1 + w2\*i2
2. h2 = w3\*i1 + w4\*i2

Activation values for Hidden Layer (Sigmoid):
1. a_h1 = σ(h1) = 1/(1 + exp(-h1))
2. a_h2 = σ(h2) = 1/(1 + exp(-h2))

Output Layer values:
1. o1 = w5\*a_h1 + w6\*a_h2
2. o2 = w7\*a_h1 + w8\*a_h2

Activation values for Output Layer (Sigmoid):
1. a_o1 = σ(o1) = 1/(1 + exp(-o1))
2. a_o2 = σ(o2) = 1/(1 + exp(-o2))

Error Calculations:
1. E1 = (1/2) \* (t1 - a_o1)^2
2. E2 = (1/2) \* (t2 - a_o2)^2
3. E_Total = E1 + E2

### Backpropagation Calcualtions
As the name suggests the final layer updation value will be calculated first and go one by one to earlier layers
1. Calculation of gradient for w5:
      - ∂E_total/∂w5 = ∂(E1+E2)/∂w5	
      - As E2 is constant wrt w5 thus only E1 remains
        - ∂E_total/∂w5 = ∂E1/∂w5
      - Simplifying equation further
        - ∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1\*∂a_o1/∂o1\*∂o1/∂w5
      - ∂E1/∂ao1 is calculated below
        - ∂E1/∂ao1 = ∂((1/2) \* (t1 - a_o1)^2)/∂a_o1 = (a_o1 - t1)			
      - ∂a_o1/∂o1 is calculated below (sigmoid differential)
        - ∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = a_o1 \* (1 - a_o1)			
      - ∂o1/∂w5 calculation	
        - ∂o1/∂w5 = a_h1	
      - Putting all pieces together we get:
        - ∂E_total/∂w5 = (a_01 - t1) \* a_o1 \* (1 - a_o1) \* a_h1	
        	
2. Similarly for weights w6, w7 & w8 the partial derivative can be calculated. Here are the equations for them respectively:
      - ∂E_total/∂w6 = (a_01 - t1) \* a_o1 \* (1 - a_o1) \* a_h2
      - ∂E_total/∂w7 = (a_02 - t2) \* a_o2 \* (1 - a_o2) \* a_h1
      - ∂E_total/∂w8 = (a_02 - t2) \* a_o2 \* (1 - a_o2) \* a_h2
      
3. Gradient for a_h1 is summation of two values as this node is connected by two routes.
      - ∂E_total/∂a_h1 = ∂E1/∂a_h1 + ∂E2/∂a_h1
      - ∂E1/∂a_h1 can be represented as below with subsequent steps:
        - ∂E1/∂a_h1 = ∂E1/∂a_o1\*∂a_o1/∂o1\*∂o1/∂a_h1
        - ∂E1/∂a_h1 = (a_o1 - t1) \* a_o1 \* (1 - a_o1) \* w5
      - Similarly ∂E2/∂a_h1 can be written as:
        - ∂E2/∂a_h1 = (a_o2 - t2) \* a_o2 \* (1 - a_o2) \* w7 
      - ∂E_total/∂a_h1 becomes:
        - ∂E_total/∂a_h1 = (a_o1 - t1) \* a_o1 \* (1 - a_o1) \* w5 + (a_o2 - t2) \* a_o2 \* (1 - a_o2) \* w7
        
4. Similarly ∂E_total/∂a_h2 is:
      - ∂E_total/∂a_h2 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8

5. Gradient for w1 can be written as:
      - ∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
      - ∂E_total/∂a_h1 is already calculated in step 3
      - ∂a_h1/∂h1 is sigmoid differential
      - ∂h1/∂w1 is equal to i1
      - Overall equation for ∂E_total/∂w1 can be represented as:
        - ∂E_total/∂w1 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * ( a_h1 * (1 - a_h1) ) * i1

6. Similarly for w2, w3 and w4, following are their gradient respectively:
      - ∂E_total/∂w2 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * ( a_h1 * (1 - a_h1) ) * i2
      - ∂E_total/∂w3 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * ( a_h2 * (1 - a_h2) ) * i1
      - ∂E_total/∂w4 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * ( a_h2 * (1 - a_h2) ) * i2

7. Once we have gradient values for all the derivatives, weights are adjusted using these gradients after multiplying them with Learning rate.

8. This process is repeated a number of times, these reretition number is the number of epochs in basic terms.

In the current example we are working with assumed input values, output values & initial weights (mentioned in the image above). By changing the number of learning rates we are trying to understand the nature of convergence.

Different Learning Rates we observed - [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]

### Loss Graphs as we change the learnign rate
Learning rate 0.1:

<img width="527" alt="Loss_lr_0 1" src="https://user-images.githubusercontent.com/118976187/212170319-e0392cd7-ed8e-4595-9ad7-2496956605ea.png">

Learning rate 0.2:

<img width="527" alt="Loss_lr_0 2" src="https://user-images.githubusercontent.com/118976187/212170340-16792cb4-176a-49b5-802b-97c2674bf3bd.png">

Learning rate 0.5:

<img width="524" alt="Loss_lr_0 5" src="https://user-images.githubusercontent.com/118976187/212170394-609db28f-b67b-4a1c-b31a-0a957e8eac90.png">

Learning rate 0.8:

<img width="527" alt="Loss_lr_0 8" src="https://user-images.githubusercontent.com/118976187/212170471-14903c2a-a502-4f23-955a-516f1353aaad.png">

Learning rate 1.0:

<img width="527" alt="Loss_lr_1" src="https://user-images.githubusercontent.com/118976187/212170507-4fa853b9-e29f-4592-97ba-fc41f3f7c11b.png">

Learning rate 2.0:

<img width="526" alt="Loss_lr_2" src="https://user-images.githubusercontent.com/118976187/212170534-c6d7709c-32b3-4b99-a199-38f18a4b068c.png">

It can be seen that for this dataset, as we are increasing learning rate loss is getting reduced rapidly. But if we keep on increasing it, loss might not converge at all!

Original [Google Sheet Link](https://docs.google.com/spreadsheets/d/1c6reSB9WeFN9p9b67E--qB4dGXEzBNdCNYK5MyVNtKo/edit?usp=sharing)

## Part-2 - Applying Multiple Concepts to achieve higher accuracy in small steps

Problem statement:
Design network fot MNIST which satisfies following conditions:
  - 99.4% validation accuracy
  - Less than 20k Parameters
  - Less than 20 Epochs
  - Have used BN, Dropout, a Fully connected layer, have used GAP. 

Following components are used in designing of netwrok:
  - 3x3 Convolutions
  - Batch Normalization
  - DropOut
  - 1x1 Convolutions
  - MaxPooling
  - Fully Connected Layer
  - GAP
  - Image Normalization

The summary of network designed:

<img width="565" alt="NN_Summary" src="https://user-images.githubusercontent.com/118976187/212171454-aeed9cc4-2ead-40b4-8808-01f1266f6f58.png">

Major points about Network designed:
- Network has total 9 layers 
- Logic used for designing layers is CRB (Convolution-Relu-Batch Normalization)
- Dropout of 0.05% is used after Batch Normalization layer
- Dropout & Batch Normalization not used after 1x1 convolution layer
- 1x1 convolution is used after two 3x3 convolutions followed by Max pooling
- Number of channels vary from 8 to 32 at different layers
- GAP is used near to last layer after convolution and a layer before fully connected layer
- Fully connected layer is last layer of network
- Log Softmax used as last layer activation function with NLL Loss 

Trainable parameters for network are 13,770 (less than 20k) and it was able to achieve ~99.3% test/validation accuracy consistently from 12th epoch onwards. Network reached maximum validation accuracy at 19th epoch to 99.38% (9938/10000). 

<img width="876" alt="Screenshot 2023-01-13 at 1 50 01 AM" src="https://user-images.githubusercontent.com/118976187/212172291-ce66d0ba-4d82-4326-a210-03b4810cac05.png">

<img width="860" alt="Screenshot 2023-01-13 at 1 50 17 AM" src="https://user-images.githubusercontent.com/118976187/212172307-45dadb4e-8992-42b1-8504-40c4ded4ce23.png">

<img width="892" alt="Screenshot 2023-01-13 at 1 50 27 AM" src="https://user-images.githubusercontent.com/118976187/212172328-223b0cd8-7a34-4e31-bc76-f4d8e41e8846.png">
