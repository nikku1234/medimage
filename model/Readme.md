This folder contains the model defenitions. 
As of now there is the defenition for the unet model and segment model.
   
   -  unet(pretrained_weights = None,input_size = (512,512,1))
        The default input size is taken as 512X512X1. This can be changed according to the user's need.
        
    - segnet(nClasses , optimizer=None , input_height=360, input_width=480, kernel = 3, filter_size = 64, pad = 1,pool_size = 2 )
        The default  optimizer=None , input_height=360, input_width=480, kernel = 3, filter_size = 64, pad = 1,pool_size = 2
 
    
