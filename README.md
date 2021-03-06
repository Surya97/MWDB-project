# Multimedia and Web Databases (Fall 2019) project

## 11K Hands : Classification and Image recommendation

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
Similarity measure used is Chi Square Statistic.
 https://link.springer.com/content/pdf/10.1007%2F978-3-540-24670-1_36.pdf
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

### Phase 2:

* **Task 1:** 
Implement a program which (a) lets the user to chose among one of the four feature models from Phase 1 and
(b) given a positive integer value, k, identifies and reports the top-k latent semantics in the corresponding vector space
using (c) one of the following techniques chosen by the user: <br/>
To run Task 1,
    ```
    $ python Phase_2.py

    Please specify the task number: 1
    Select specify the test folder path: <folder path>
    1. CM
    2. LBP
    3. HOG
    4. SIFT
    Select the decomposition: <Decomposition Name>
    1. PCA
    2. SCD
    3. NMF
    4. LDA 
    Enter the number of latent features to consider: <latent number>
        
    ```
* **Task 2:**
Implement a program which (a) lets the user to chose among one of the four feature models and (b) given the
top-k latent semantics for that feature model, created using (c) a dimensionality reduction technique chosen by the user,
and given (d) an image ID, the system identifies the most related m images in the k-dimensional latent space (list also
the matching scores).<br/>
To run Task 2,
    ```
    $ python Phase_1.py

    Please specify the task number: 3
    Select specify the test folder path: <folder path>
    1. CM
    2. LBP
    3. HOG
    4. SIFT
    Select the Model: <Model Name>
    1. PCA
    2. SCD
    3. NMF
    4. LDA 
    Select the Decomposition : <Decomposition Name>
    Please specify the test image file name:<File Name>
    Please specify the number of components:<num_components>
    Please specify the value of m:<m_value> 
    ```

* **Task 3:**
Implement a program which (a) lets the user to chose among one of the four feature models and (b) given one of
the labels,
– left-hand,
– right-hand,
– dorsal,
– palmar,
– with accessories,
– without accessories,
– male, or
– female
identifies (and lists) k latent semantics for images with the corresponding metadata using (c) one of the following techniques chosen by the user: <br/>
To run Task 3,
    ```
    $ python Phase_2.py

    Please specify the task number: 3
    1. CM
    2. LBP
    3. HOG
    4. SIFT
    Select the Model: <Model Name>
    1. PCA
    2. SCD
    3. NMF
    4. LDA 
    Select the Decomposition : <Decomposition Name>
    1.Left-Hand
    2.Right-Hand
    3.Dorsal
    4.Palmar
    5.With accessories
    6.Without accessories
    7.Male
    8.Female
    
    Please choose an option: <label_number>
    Please specify the number of components:<num_components>
    
                                          
    ```
    
 * **Task 4:**
Implement a program which (a) lets the user to chose among one of the four feature models and (b) one of the
four techniques (PCA, SVD, NMF, or LDA) and (c) given the k latent semantics associated with one of the labels,
– left-hand,
– right-hand,
– dorsal,
– palmar,
– with accessories,
– without accessories,
– male, or
– female
and (d) given an image ID, identifies the most related m images using these k latent semantics (list also the matching
scores). <br/>
To run Task 4,
    ```
    $ python Phase_2.py
    Please specify the task number: 4
    1. CM
    2. LBP
    3. HOG
    4. SIFT
    Select the Model: <Model Name>
    1. PCA
    2. SCD
    3. NMF
    4. LDA 
    Select the Decomposition : <Decomposition Name>
    1.Left-Hand
    2.Right-Hand
    3.Dorsal
    4.Palmar
    5.With accessories
    6.Without accessories
    7.Male
    8.Female
    
    Please choose an option: <label_number>
    Please specify the number of components:<num_components>
    Please specify the test image file name:<File Name>
    Please specify the value of m:<m_value> 
    ```   
* **Task 5:**
Implement a program which (a) lets the user to chose among one of the four feature models and (b) one of the
four techniques (PCA, SVD, NMF, or LDA) and (c) given the k latent semantics associated with one of the labels,
– left-hand,
– right-hand,
– dorsal,
– palmar,
– with accessories,
– without accessories,
– male, and
– female
and (d) an unlabeled image ID, the system labels it as
– left-hand vs right-hand,
– dorsal vs palmar
– with accessories vs. without accessories
– male vs. female <br/>
To run Task 5,
    ```
    $ python Phase_2.py

    Please specify the task number: 5
    1. CM
    2. LBP
    3. HOG
    4. SIFT
    Select the Model: <Model Name>
    1. PCA
    2. SCD
    3. NMF
    4. LDA 
    Select the Decomposition : <Decomposition Name>
    1.Left-Hand
    2.Right-Hand
    3.Dorsal
    4.Palmar
    5.With accessories
    6.Without accessories
    7.Male
    8.Female
    
    Please choose an option: <label_number>
    Please specify the test image file name:<File Name>
                                              
    ```
* **Task 6:** 
Implement a program which given ( a subject ID, identifies and visualizes the most related 3 subjects (you are
free the use any feature model and latent semantics) <br/>
To run Task 6,
    ```
    $ python Phase_2.py

    Please specify the task number: 6
    Please input the subject Id: <subject_id>     
    '''
    
* **Task 7:** 
Implement a program which, given a value k,
– creates a subject-subject similarity matrix,
– performs NMF on this subject-subject similarity matrix, and– reports the top-k latent semantics. <br/>
To run Task 7,
    ```
    $ python Phase_2.py

    Enter the number of latent features to consider:<latent_num>

* **Task 8:** 
Implement a program which, given a value k,
– creates a binary image-metadata matrix,
– performs NMF on this image-metadata matrix, and
– reports
∗ top-k latent semantics in the image-space.
∗ top-k latent semantics in the metadata-space.
Each latent semantic should be presented in decreasing order of weights.
To run Task 8,
    ```
    $ python Phase_2.py

    Enter the number of latent features to consider:<latent_num>
