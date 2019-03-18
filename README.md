# deep_hockey
Play around with deep learning. Basic idea is to use ML to analyze kids hockey practices.

# Deep learning algos and stuff
Use Python Anaconda, Tensorflow and OpenCV with help from [ivangrov](https://github.com/ivangrov/YOLOv3-Series/blob/master/%5Bpart%201%5DYOLOv3_with_OpenCV/OD.py).

We start by employing the YOLO-v3 deep learning algo. It looks nice.

# Game plan
## Part 1 - Get friends with YOLO
Need first to get familiar with YOLO and see what it can manage on its own, without any training.

### First stab at it
Very nice indeed. YOLO with the recommended weights picks out players without problems. At Threshold 0.1 it picks out lots of stuff, mistakes legs and hockey sticks for skis.
Also it finds some other stuff. That's not a problem as long as it reliably picks out the players.

![alt text][stab-1]

[stab-1]: https://github.com/nwesar/deep_hockey/raw/master/data/classify-thres-0.3.jpg "1st classification test, Yolov2, Threshold 0.3"


## Part 2 - Analyze the learned data
To begin with, it would be interesting to just figure out how efficient the practices are: can we calculate
how large part of the practice the kids spend moving and standing still (waiting, instruction, etc)?

## Part 3 - Train YOLO to classify new objects
With the initial weights, YOLO would successfully classify players as "Person". That is good enough.
But we need it to recognize also other things that are interesting for our upcoming analysis:
* pucks
* the rink itself?
* E/N-zones
* faceoff points for distance reference to use when calculating skating distance, speed, etc?

Cheesus, this is going to be a long one...

## Part N - Complete analysis
Envisioned "analysis" workflow:

0) Install some cameras at hockey rink
1) Record kids hockey practice video (fancy automation stuf goes here)
2) Use YOLO to detect players, pucks, etc in video. Store in some "workable format"
3) Calculate analytics from practice, e.g. , "puck handling time", etc.
4) Present analysis on web page or similar
