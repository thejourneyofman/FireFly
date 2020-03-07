Firefly Graph – Lockdown Densest Sub Graph Dynamically
============
Firefly Graph is an optimized data structure using the graph theory to identify the "densest sub graph" for the management of the space-time data.

## Demo
![demo](https://github.com/thejourneyofman/firefly/blob/master/images/demo.gif)

Motivation
==========
In many cases of AI and big data adventures, we have to manage sparse data in both time series and space sequence where most of (X,Y,T) in a three-dimensional matrix is zero or practically meaningless; only few dense zones [(Xi,Yj, Tk)~(Xi+r,Yj+s, Tk+t)] have to be extracted and conducted in analysis. One use case is the object detection in a research field of computer vision, for instance, a self-driving system where algorithms, sensors or computing power are not necessarily out of breath to all the static/background objects, like skies, trees or houses which we have every reason to decouple from the active moving objects which deserve better to our computing power. Two examples, a person crossing road or a light turning red are what we have to ensure no loss of detection. To purify the sparse space-time data will reduce the overall cost of computing power or trade the same power with a better performance. Another use case is the financial transaction supervision in pursuit of further coordinated KYC/AML processes for higher efficiency. To be headed for this goal, we put forward suggestions to reduce the data scale from dealing with all the historical transaction logs in case that the T(ime) axis is always continuous and (X, Y) grid has a large size. To lock on outliers in the transactions by user-defined rules has been proved to be an effective way but as a side effect it also poses false-positive or false-negative risks. In general, how to trade data integrity with the performance looks more like a realistic problem. This project aims at challenging an optimized data structure that uses graph and geometry theories to find the densest subsets of the space-time data, to make further searching, clustering or classification more flexible and effective.

[Go camping! -- A live Demo](https://fireflycruise.herokuapp.com)

Concept
==========
In the sample movies, hundreds of dancing fireflies are being captured by a hi-definition camera. Let’s set up a question, how can we identify the most active/densest zones where the brightest, largest or speediest fireflies danced in? We introduce three concepts into our graph structures, “footprint”, “resort” and “cruise zone”. Each firefly has many “footprints”, starting from being captured, growing larger and brighter, then fading out of our sight. Their tracked and gathered locations are defined as “resorts” which can be composed of many footprints. In our graph, each node represents a footprint of fireflies at a certain point in timeline. The goal of this project is to extract and identify the densest resorts in rectangles that can lockdown the most active zones where fireflies have toured, in other words, it will restructure the entire sparse frames from 3D (X,Y,T) to a multilayered and graded graph structure and each layer represents a sub graph, in this case, which is defined as a “cruise zone”.

**Prerequisites:**

- [Python 3](https://www.python.org/),  [numpy](http://www.numpy.org/),

**An interactive Live UI to generate a FireFly Graph**

#####  HOW TO USE IT
- 1, Click [A Live DEMO](https://fireflycruise.herokuapp.com).
- 2, Select a test scenario from the dropdown list, each scenario is a short video of fireflies captured.
- 3, Play it and Click "Generate Graph".

Available Classes
-----------------

There are three classes in `Firefly Graph`:

- `Footprint` - Returns a class of a single node named as Footprint in the Graph. Each node represents a footprint of fireflies at a certain point in timeline.
- `Resort` - Returns a class of a resort or a group of footprints in the Graph. Each resort represents a group of footprints of fireflies.
- `Cruise` - Returns a class of a graph. Each Cruise represents a group of resorts where fireflies have toured.

LICENSE
=======

The code which makes up this project is licensed under the MIT/X11 license.

Copyright (c) 2020 Chao (Chase) Xu
