### Phase 3:

* **Task 2:** 
Task 2: Implement a program which, given a folder with dorsal/palmar labeled images and for a user supplied c,
– computes c clusters associated with dorsal-hand images (visualize the resulting image clusters),
– computes c clusters associated with palmar-hand images (visualize the resulting image clusters),
  and, given a folder with unlabeled images, the system labels them as
– dorsal-hand vs palmar-hand using only descriptors of these clusters.
   To run Task 2,
    ```
    $ python Phase_3.py

    Please specify the task number: 2
    Enter the number of clusters: <number_of_clusters>
    Enter labelled dataset path: <folder path>
    Enter Unlabelled dataset path : <folder path> 
    ```
  
* **Task 3:** 
Implement a program which, given a value k, creates an image-image similarity graph, such that from each
image, there are k outgoing edges to k most similar/related images to it. Given 3 user specified imageids on the graph,
the program identifies and visualizes K most dominant images using Personalized Page Rank (PPR) for a user supplied
K.
    To run Task 3,
    ```
    $ python Phase_3.py

    Please specify the task number : 3
    Enter folder path : <folder - path>
    Enter 3 ImageIds: <images_ids>
    Enter the number of outgoing edges:<number>
    ```
  
* **Task 4:** 
Implement a program which, given a folder with dorsal/palmar labeled images,– creates an SVM classifer,
– creates a decision-tree classifier,
– creates a PPR based clasifier,
 and, given a folder with unlabeled images, the system labels them as
– dorsal-hand vs palmar-hand
using the classifier selected by the user.
    To run Task 4,
    ```
    $ python Phase_3.py

    Please specify the task number : 4
    1.SVM
    2.DT
    3.PPR
    Select the Classifier: <classifier>
    Enter labelled_dataset_path : <folder - path>
    Enter unlabelled_dataset_path : <folder - path>
    ```
* **Task 5:** 
5a: Implement a Locality Sensitive Hashing (LSH) tool (for Euclidean distance) which takes as input (a) the number
of layers, L, (b) the number of hashes per layer, k, and (c) a set of vectors as input and creates an in-memory index
structure containing the given set of vectors.
– 5b: Implement a similar image search algorithm using this index structure and a visual model function of your
choice (the combined visual model must have at least 256 dimensions): for a given query image and integer t,
visualizes the t most similar images (also outputs the numbers of unique and overall number of images considered).    
To run Task 5,
    ```
    $ python Phase_3.py

    Please specify the task number : 5
    Enter the number of Layers: <number>
    Enter the number of Hashes per layer: <number>
    Enter the ImageId: <image_id>
    Enter the value of t: <task_number>
    ```
* **Task 6:** 
Let us consider the label set “Relevant (R)” and “Irrelevant (I)”. Implement
– an SVM based relevance feedback system,
– a decision-tree based relevance feedback system,
– a PPR-based relevance feedback system,
– a probabilistic relevance feedback system
which enable the user to label some of the results returned by 5b as relevant or irrelevant and then return a new set of
ranked results, relying on the feedback system selected by the user, either by revising the query or by re-ordering the
existing results.    
To run Task 6,
    ```
    $ python Phase_3.py

    Please specify the task number : 6
    Number of Images you would like to label as Relevant: <number>
    Number of Images you would like to label as Irrelevant: <number>
    <You will get the list of images>
    Enter the ImageNumber to Label as Relevant:<number>
    Enter the ImageNumber to Label as Irrelevant:<number>
    ```
