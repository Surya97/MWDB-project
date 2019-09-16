# Multimedia and Web Databases (Fall 2019) project

## 11K Hands :Gender Recognition and Biometric Identification using large dataset of hand images

### Phase 1:

**Note:** The CLI for the Phase 1 is inside the folder Phase 1, named Phase_1.py 

* **Task 0:** In this project, we will use the data sets associated with the following publication:
Mahmoud Afifi. “11K Hands: Gender recognition and biometric identification using a large dataset of hand
images.” M. Multimed Tools Appl (2019) 78: 20835.
The data sets are available at https://sites.google.com/view/11khands

* **Task 1:** 
Implement a program which, given an image ID and one of the following models, extracts and prints (in a human
readable form) the corresponding feature descriptors:<br />
    a\. **Local binary patterns, LBP100x100:** Split the image into 100x100 windows, compute LBP features for each window, and concatenate these to obtain a unified feature descriptor.
Note that LBP is computed on gray scale images thus, need to first convert the images to gray scale. 
    ```text
    Parameters used:
    radius: 2
    Number of circularly symmetric neighbour set points: 8
    ```
  b\. **Histograms of oriented gradients, HOG:** Convert the images to gray scale and then down-sample images 1-per-10 (rows and columns).
    ```text
    Parameters used:
    Number of orientation bins: 9
    Pixels per cell: 8
    Cells per block: 2
    ```
    To run Task 1,
    ```
    $ python Phase_1.py

    Please specify the task number: 1
    1.LBP
    2.HOG
    Select model: <Model Name>
    Please specify the test image file name: <Test Image File name>
    ```
* **Task 2:**
Implement a program which, given a folder with images, extracts and stores feature descriptors for all the images
in the folder.<br/>
To run Task 2,
    ```
    $ python Phase_1.py

    Please specify the task number: 2
    1.LBP
    2.HOG
    Select model: <Model Name>
    Please specify test folder path: <Test Image Folder path>
    ```

* **Task 3:**
Implement a program which, given an image ID, a model, and a value “k”, returns and visualizes the most
similar k images based on the corresponding visual descriptors. For each match, also list the overall matching score.<br/>
To run Task 3,
    ```
    $ python Phase_1.py

    Please specify the task number: 3
    1.LBP
    2.HOG
    Select model: <Model Name>
    Please specify the test image file name: <Test Image File name>
    Please specify k: <Number of similar images to visualize>
    Please specify test folder path: <Test Image Folder path>
    ```
