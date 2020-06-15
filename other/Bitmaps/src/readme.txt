Model.java represents the PDP model, and ModelTester.java is the main class which takes in user input to create and run the model.

ConfigScanner.java is a custom scanner to read over a config file and provide the Model class with the specified Model configuration.

BMPUtil.java, based on the provided DibDump class (not in this archive), takes in a BMP file and parses its header and pel data. 
It can read and parse BMP files, can scale and print the pel values to a file, read scaled pel values from a file, and print an output BMP.
While BMPUtil is not the main class of the project, it has a main method which is useful for reading in bitmaps and outputting their pel values into a pel file. 
The pel files are required for the config file to determine the test case inputs and outputs, but running a model with a bitmap's pel values can also be done through the ModelTester's command line prompts by passing in the file path to the bitmap itself.

Example Config files are provided in the models folder. Bitmaps that can be used for training models are in the bmps folder. 
The two BMPModels in the models folder will use the pel values of these bitmaps, already in pel files in the top level directory, for training.