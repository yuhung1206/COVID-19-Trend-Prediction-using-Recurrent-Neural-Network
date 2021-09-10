# COVID-19-Trend-Prediction-using-Recurrent-Neural-Network
Predict whether the number of confirmed cases will increase or not with the use of RNN, LSTM &amp; GRU.

Observe the sequences of comfirmed cases during 2020-1-22 ~ 2020-4-12 from 185 countries.  
The gloabl data of comfired cases please refer to -> https://ourworldindata.org/coronavirus  

  ![image](https://user-images.githubusercontent.com/78803926/132695426-95a8f4cd-b9e2-40ea-bd24-66a3b5f4d9af.png)  
  
  
## Execution & Overall Structure of system  
 1. **Sequence Preprocessing** : find high correlated countries & prepared segments for sequence modelling  
    ```
    python3 Preprocess.py
    ```
 2. **RNN** for trend prediction : training the model with **Torch** package.  
    Visualization on a world map by **"pygal"** packag  
     ```
    python3 RNN.py
    ```  
    
     
## Sequence Preprocessing  
  1. Compute to **correlation coefficient (CC)** between 2 countries.  
     ![image](https://user-images.githubusercontent.com/78803926/132826430-d0208042-35dc-497c-bde2-730d73fd9e42.png)
  
  2. Add the pair of countries to set C if their CC higher than 0.7 (highly-correlated)  
     ![image](https://user-images.githubusercontent.com/78803926/132838262-b0dca023-09bb-46bb-859d-a272e4f362b7.png)

  4. Generate the data pair (segment,lebel) for modelling from set C  
    
    
    
## RNN for Classification  
 
  - **RNN with 2 layers & dropout**  
    In this project, dropout is selected to aviod overfitting problem.  
    ![image](https://user-images.githubusercontent.com/78803926/132838592-0dd30435-9c5d-461d-8aaa-ffd64927c2df.png)  
      
    ![image](https://user-images.githubusercontent.com/78803926/132838897-fa345ff7-64c5-48f6-b665-75130e513138.png) 
      
      
## Visualization of prediction on a World Map  
![image](https://user-images.githubusercontent.com/78803926/132839458-6d5ea0f0-a40d-4f50-ac82-68ad6cc7efc9.png)



