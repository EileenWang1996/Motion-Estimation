# Motion-Estimation

This code performs motion estimation on an input video using the macroblock matching method. The result is an output video which
contains dots on parts of each frame, indicating that the block in that part of the frame has a large displacement
value. To run the code in a jupyter notebook, you must first ensure that you have Ananconda and Python installed on your machine. 

To apply the motion estimation algorithm on a video, you must first choose a video of your choice. Then, change the 'path_to_vid'
variable in the third cell of the notebook to the path of where your video is located. You will also need to change the variable
'frame_save_path' to the destination of where you want to save the edited frames. Additionally, change the path to where you would 
like to save the output video by modifying the 'output_vid_path' variable. Finally, simply run the code and see what happens! 
