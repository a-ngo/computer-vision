# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

Mid-term project: udacity 2D feature tracking


## Performance evaluation
The different combinations of detectors and descriptors are compared in [method_comparison](method_comparison.ods).

The top 3 combinations are chosen based on the execution time and the ratio of number of matched keypoints to the number of detected keypoints (the higher the ratio, the better):
1. Orb detector and Brief descriptor
    - very fast (0.047s)
    - high ratio (0.54)

2. Harris detector and Brief descriptor
    - very fast (0.093s)
    - good ratio (0.49)

3. Harris detector and Sift descriptor
    - fast (0.151s)
    - high ratio (0.56)
