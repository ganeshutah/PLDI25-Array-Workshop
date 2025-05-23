# PLDI25-Array-Workshop

* To reproduce the results in this paper, you'll need GPU-FPX's old version with the hang bug fix (for Sec 3.1-3.3)
   - Follow instructions on https://github.com/LLNL/GPU-FPX to install
     and build GPU-FPX. Then try out the examples in this paper as
     described below.
   - The version without the hang is in the nvbit-1.7.5 branch of https://github.com/LLNL/GPU-FPX.
* To try out the new Tensor-Core version of GPU-FPX that has merged
  the functionality into one tool called "detector.so", do this:
   - Follow the instructions on https://github.com/LLNL/GPU-FPX
   - ...change/add the relevant things...

* To reproduce the SRU results of Sec 3.1, please do the following:
   - As instructed in https://github.com/asappresearch/sru/, do pip install sru
   - Then follow the script fully given in the paper.
   
* To reproduce the results in Sec 3.2, please do the following:
   - The version that does not have any issues is main.cu (given here
     on this Github)
   - The version that has the issues is main1.cu (given here on this
     Github)
   - Whichever you are running, compile the file as follows: for
      main.cu, it is as follows:
   - nvcc -O3 -arch=sm_75 -lineinfo -o cuszp main.cu
   - Here, set  arch=... version before running to match your GPU)
   - Then run LD_PRELOAD=detector.so ./cuszp  
   - change 'detector' to 'analyzer' to conduct analysis
   - Change to nvcc -O3 -arch=sm_75 -lineinfo -o cuszp1 main1.cu
   - Then run LD_PRELOAD=detector.so ./cuszp1 to see the bug reported
     in the paper
   - main1.cu differs from main.cu in that its tests have higher coverage

   - Each run of cuszp or cuszp1 will generate a test data file and
     save the compressed data into the file test_data.cuszp.bin 
	 
* To reproduce the results in Sec 3.3 involving PyBlaz, do the
  following
    - First install PyBlaz as instructed here:
    - Next, you need to run script3.py which is included in this
      Github
	  


