# Motion Estimation with Macroblock Matching

This code performs motion estimation on an input video using the macroblock matching method (read more about macroblock matching here: https://en.wikipedia.org/wiki/Block-matching_algorithm). 

The result is an output video which contains dots on parts of each frame, indicating that the block in that part of the frame has a large displacement value. To run the code in a jupyter notebook, you must first ensure that you have Ananconda and Python installed on your machine. 

To apply the motion estimation algorithm on a video, you must first choose a video of your choice. Then, change the 'path_to_vid'
variable in the third cell of the notebook to the path of where your video is located. You will also need to change the variable
'frame_save_path' to the destination of where you want to save the edited frames. Additionally, change the path to where you would 
like to save the output video by modifying the 'output_vid_path' variable. Finally, simply run the code and see what happens! 

Examples of output frames: 

![frame_example](https://user-images.githubusercontent.com/35329219/57976954-2c31f080-7a30-11e9-9f53-9b051529bd2f.jpg)

![frame_example2](https://user-images.githubusercontent.com/35329219/57976958-44097480-7a30-11e9-9df2-75f81a5095a6.jpg)
