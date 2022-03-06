import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.NoSuchElementException;
import java.util.Scanner;

/**
 * Scans a PDP model config file for a PDP Model and stores the values to construct, run, and train the model.
 * 
 * For an example config file, see TemplateConfig.txt in the 'models' folder.
 * 
 * The ConfigScanner class has two constructors so that a ConfigScanner can be created using a String file 
 * path or a File object. Both constructors make a call to the readConfig method described below.
 * 
 * The readConfig method creates a java.util.Scanner object at the beginning of the file and parses every 
 * part of the file. It uses four private methods to parse each section of the config file: 
 * 
 *    The readModelConfig method parses the 'Model Format' section in the config file.
 *    The readModelWeights method parses the 'Weights' section in the config file.
 *    The readModelTestCases method parses the 'Test Cases' section in the config file. If the test cases are 
 *       listed in external files and not in the config file itself, the readTestCaseFromFiles method is used
 *       to fill the testCases array with values from the test case files.
 *    The readModelTrainingConfig method parses the 'Training Config' section in the config file.
 * 
 * The writeConfigToFile method is used for updating the config file based on the current values of the 
 * ConfigScanner's fields, such as when the model's weights are changed after training and need to be saved.
 * 
 * The ConfigScanner uses three helper methods, readNextString, readNextInt, and readNextDouble, to navigate 
 * through the config file and parse certain tokens while skipping over others of different types.
 * 
 * @author Jatin Kohli
 * @since 9/20/19
 */
public class ConfigScanner 
{
   /*
    * Tokens in the Config File that are used by the scanner to skip past text and find important values.
    * These must be in the correct places (see 'TemplateConfig.txt') for the ConfigScanner to work properly
    */
   private static final String FIRST_FORMAT_TOKEN = "Format:";    //Start of the 'Model Format' section
   private static final String THRESHOLD_TOKEN = "'sigmoid'):";   //Precedes 'Threshold Function' token
   public static final String LINEAR_TOKEN = "linear";            //Value of a linear 'Threshold function' token
   public static final String SIGMOID_TOKEN = "sigmoid";          //Value of a sigmoid 'threshold function' token
   private static final String FIRST_WEIGHTS_TOKEN = "val";       //Start of the 'Weights' section
   private static final String FIRST_TEST_CASES_TOKEN = "Cases:"; //Start of the 'Test Cases' section
   private static final String PRECEDING_CASE_TOKEN = "Actual:";  //Precedes the first actual Test Case
   private static final String FIRST_TRAINING_TOKEN = "Config:";  //Start of the 'Training Config' section

   private static final String INDENT = "   ";

   public static final int TEST_CASE_INPUT_INDEX = 0;  //Index where a test case's inputs are located
   public static final int TEST_CASE_OUTPUT_INDEX = 1; //Index where test case outputs are located

   private String[] inputFiles;
   private String[] outputFiles;

   private Scanner scanner;
   private File configFile;

   public int inputNodes;
   public int numHiddenLayers;
   public int[] hiddenLayerNodes;
   public int outputNodes;
   public String thresholdFunction;

   public int maxLayerSize;
   public int numTestCases;

   public double[][][] weights;
   public double[][][] testCases;

   public double defaultLambda;
   public double deltaLambda;
   public double minLambda;
   public int maxIterations;
   public double maxAcceptableError;
   public double minRandom;
   public double maxRandom;
   public int savePeriod;

   /**
    * Creates a ConfigScanner over a file at the specified path
    *
    * @param configFilePath 
    *       The String pathname denoting the file to be read by the ConfigScanner
    */
   public ConfigScanner(String configFilePath) 
   {
      this(new File(configFilePath));
   }

   /**
    * Creates a ConfigScanner over a file
    *
    * @param configFile 
    *       The configuration file, following the structure specified of the file called 
    *       'TemplateConfig.txt' in the models folder
    */
   public ConfigScanner(File configFile) 
   {
      this.configFile = configFile;

      readConfig();
   }

   /**
    * Reads all parts of the config file and stores the relevant values
    */
   public void readConfig()
   {
      try 
      {
         scanner = new Scanner(configFile);
      } 
      catch (FileNotFoundException e) 
      {
         e.printStackTrace();
      }
      
      readModelConfig();
      readModelWeights();
      readModelTestCases();
      readTrainingConfig();
   } //public void readConfig()

   /**
    * Reads the 'Model Format' section and stores the relevant values
    */
   private void readModelConfig()
   {
      readNextString(FIRST_FORMAT_TOKEN); //Set scanner to correct position to read model format

      inputNodes = readNextInt();

      numHiddenLayers = readNextInt();

      hiddenLayerNodes = new int[numHiddenLayers];

      for (int n = 0; n < numHiddenLayers; n++) //Determine how many nodes are in each hidden layer
         hiddenLayerNodes[n] = readNextInt();
      
      outputNodes = readNextInt();

      maxLayerSize = Math.max(inputNodes, outputNodes);

      for (int hiddenLayerSize : hiddenLayerNodes)
         maxLayerSize = Math.max(maxLayerSize, hiddenLayerSize);

      readNextString(THRESHOLD_TOKEN); //Set scanner to correct position to read threshold function

      thresholdFunction = scanner.next();

      if (!thresholdFunction.equals(LINEAR_TOKEN) && !thresholdFunction.equals(SIGMOID_TOKEN))
      {
         throw new IllegalArgumentException("Threshold Function Type from the Config File " + 
               "was invalid (not 'linear' or 'sigmoid')"
         );
      }
   } //private void readModelConfig()
   
   /**
    * Reads the Weights section and stores the relevant values
    */
   private void readModelWeights()
   {  
      weights = new double[numHiddenLayers + 1][maxLayerSize][maxLayerSize];

      readNextString(FIRST_WEIGHTS_TOKEN); //Set scanner to correct position to read weights
      
      while (scanner.hasNextInt()) //Loop while there are unread weights
      {
         int m = scanner.nextInt();
         int j = scanner.nextInt();
         int i = scanner.nextInt();
         double val = scanner.nextDouble();

         if (m < numHiddenLayers + 1 && j < maxLayerSize && i < maxLayerSize) //If weight is valid
            weights[m][j][i] = val;
      }
   } //private void readModelWeights()

   /**
    * Scans the Test Cases section and stores the relevant values
    */
   private void readModelTestCases()
   {
      readNextString(FIRST_TEST_CASES_TOKEN); //Set scanner to correct position to read test cases

      numTestCases = readNextInt();

      //For each test case, there are two double arrays: one for inputs and one for desired outputs
      testCases = new double[numTestCases][2][]; 

      readNextString(PRECEDING_CASE_TOKEN); //Set scanner to correct position to read test cases

      scanner.next(); //read past '{' token

      boolean isTestsInFile = !scanner.hasNextDouble(); //If next token is not a double, it must be a file path
      
      if (isTestsInFile)
      {
         inputFiles = new String[numTestCases];
         outputFiles = new String[numTestCases]; 

         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            inputFiles[testCase] = scanner.next(); //input file path

            scanner.next(); //read past '},' token in config file
            scanner.next(); //read past '{' token in config file

            outputFiles[testCase] = scanner.next(); //output file path

            readTestCaseFromFiles(testCase, inputFiles[testCase], outputFiles[testCase]);

            scanner.next(); //read past '}' token

            if (testCase != numTestCases - 1) //If the current test case is not the last
               scanner.next(); //read past '{' token on the next line (with the next test case)
         } //for (int testCase = 0; testCase < numTestCases; testCase++)
      } //if (isTestsInFile)
      else //Test cases inputs and outputs are given in the Config File, not a seperate file
      {
         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            for (int ioIndex = 0; ioIndex <= 1; ioIndex++) //ioIndex == 0 for input(i) and 1 for output(o)
            {
               int size;

               if (ioIndex == TEST_CASE_INPUT_INDEX) 
                  size = inputNodes;
               else //ioIndex == TEST_CASE_OUTPUT_INDEX
                  size = outputNodes;

               testCases[testCase][ioIndex] = new double[size];

               for (int testValue = 0; testValue < size; testValue++)
               {
                  try
                  {
                     testCases[testCase][ioIndex][testValue] = readNextDouble();
                  } 
                  catch (Exception e)
                  {
                     throw new NoSuchElementException(
                           "Test Case not found, make sure your test case inputs/outputs are the right " + 
                           "size and the specified number of test cases in the config file is correct."
                     );
                  }
               } //for (int testValue = 0; testValue < size; testValue++)
            } //for (int ioIndex = 0; ioIndex <= 1; ioIndex++)
         } //for (int testCase = 0; testCase < numTestCases; testCase++)
      } //else
   }//private void readModelTestCases()

   /**
    * Reads the values for a test case and stores it in testCases at index testCase.
    * The test case data can be retrieved from testCases[testCase].
    *
    * @param testCase 
    *       The integer index of the testCases array to store the test case data in
    *
    * @param inputFile 
    *       The file containing activation values for each input node.
    *       The file must contain as many integer or double activation values as there are
    *       nodes in the activation layer. 
    *       The file must contain nothing but the integer or double activation values.
    *
    * @param outputFile 
    *       The file containing the desired output values when the model is run with 
    *       the inputs from the input file. The file must contain as many integer or double 
    *       output values as there are nodes in the output layer.
    *       The file must contain nothing but the integer or double desired output values.
    */
   private void readTestCaseFromFiles(int testCase, String inputFile, String outputFile)
   {
      double[] testInputs = new double[inputNodes];
      double[] testOutputs = new double[outputNodes];

      Scanner inputFileScanner = null;
      Scanner outputFileScanner = null;

      try 
      {
         inputFileScanner = new Scanner(new File(inputFile));
         outputFileScanner = new Scanner(new File(outputFile));
      } 
      catch (FileNotFoundException e) 
      {
         System.err.println("Input or Output files not found\r\n" + e.getStackTrace());
      }

      try
      {
         for (int numInput = 0; numInput < inputNodes; numInput++)
         {
            testInputs[numInput] = inputFileScanner.nextDouble();
         }

         for (int numOutput = 0; numOutput < outputNodes; numOutput++)
         {
            testOutputs[numOutput] = outputFileScanner.nextDouble();
         }
      } //try
      catch (NoSuchElementException e)
      {
         System.err.println(
               "Test Case Files did not contain enough activation or output values for the model\r\n" + 
               e.getStackTrace()
         );
      }

      testCases[testCase][TEST_CASE_INPUT_INDEX] = testInputs;
      testCases[testCase][TEST_CASE_OUTPUT_INDEX] = testOutputs;
   } //private void readTestCaseFromFiles(int testCase, String inputFile, String outputFile)

   /**
    * Scans the Training Config section and stores the relevant values
    */
   private void readTrainingConfig()
   {
      readNextString(FIRST_TRAINING_TOKEN);

      defaultLambda = readNextDouble();
      deltaLambda = readNextDouble();
      minLambda = readNextDouble();
      maxIterations = readNextInt();
      maxAcceptableError = readNextDouble();
      minRandom = readNextDouble();
      maxRandom = readNextDouble();
      savePeriod = readNextInt();
   } //private void readTrainingConfig()

   /**
    * Writes the Model's current configuration to the configuration file, 
    * overwriting the file's current contents.
    */
   public void writeConfigToFile()
   {
      String header = 
         "Change numbers to desired values \r\n" +
         "Be very careful when modifying or adding non-numerical text \r\n\r\n";
      
      String modelFormatStr = 
            "Model Format: \r\n" +
            INDENT + "Input Size: " + inputNodes + "\r\n" +
            INDENT + "Number of hidden layers: " + numHiddenLayers + "\r\n" +
            INDENT + "Hidden Layer Sizes: { ";

      for (int layer : hiddenLayerNodes)
         modelFormatStr += layer + " ";
      
      modelFormatStr += "} \r\n" +
            INDENT + "Output Size: " + outputNodes + "\r\n";

      modelFormatStr += INDENT + "Threshold Function (Must be 'linear' or 'sigmoid'): " + thresholdFunction + 
            "\r\n\r\n";

      String weightStr = "Weights: \r\n" +
            INDENT + "m j i val \r\n";

      PrintStream printer = null;

      try 
      {
         printer = new PrintStream(configFile);
      } 
      catch (Exception e) 
      {
         e.printStackTrace();
      }

      printer.print(header + modelFormatStr + weightStr);

      double weight;

      for (int m = 0; m < weights.length; m++)
      {
         for (int j = 0; j < weights[m].length; j++)
         {  
            for (int i = 0; i < weights[m][j].length; i++)
            {
               weight = weights[m][j][i];

               if (weight != 0.0)
                  printer.println(INDENT + m + " " + j + " " + i + " " + weights[m][j][i]);
            }
         }
      }

      String caseStr = "\r\n" + "Test Cases: \r\n" +
            INDENT + "Num Cases: " + numTestCases + " \r\n\r\n" +
            INDENT + "Example: \r\n" +
            INDENT + INDENT + "{ input1 input2 ... }, { output1 output2 ... } \r\n\r\n" + 
            INDENT + "Actual: \r\n";
      
      if (inputFiles == null || outputFiles == null) //If the test case values were not specified in files
      {
         for (int numTestCase = 0; numTestCase < testCases.length; numTestCase++)
         {
            caseStr += INDENT + INDENT;

            for (int ioIndex = 0; ioIndex < testCases[numTestCase].length; ioIndex++)
            {
               caseStr += "{ ";

               for (int value = 0; value < testCases[numTestCase][ioIndex].length; value++)
               {
                  caseStr += testCases[numTestCase][ioIndex][value] + " ";
               }

               caseStr += "}";
               
               if (ioIndex == TEST_CASE_INPUT_INDEX)
                  caseStr += ", ";
               else //ioIndex == TEST_CASE_OUTPUT_INDEX
                  caseStr += "\r\n";
            } //for for (int ioIndex = 0; ioIndex < testCases[numTestCase].length; ioIndex++)
         } //for (int numTestCase = 0; numTestCase < testCases.length; numTestCase++)
      } //if (inputFiles == null || outputFiles == null)
      else
      {
         for (int numTestCase = 0; numTestCase < testCases.length; numTestCase++)
         {
            caseStr += INDENT + INDENT + 
                  "{ " + inputFiles[numTestCase] + " }, { " + outputFiles[numTestCase] + " } \r\n";
         }
      }

      String trainingStr = "\r\n" +
            "Training Config: \r\n" +
            INDENT + "Initial Learning Factor: " + defaultLambda + "\r\n" +
            INDENT + "Learning Factor Modifier: " + deltaLambda + "\r\n" +
            INDENT + "Learning Factor Minimum: " + minLambda + "\r\n" +
            INDENT + "Maximum Number of Iterations: " + maxIterations + "\r\n" +
            INDENT + "Maximum Acceptable Error (for all test case to have): " + maxAcceptableError + "\r\n" +
            INDENT + "Minimum Value for Randomized Weights: " + minRandom + "\r\n" +
            INDENT + "Maximum Value for Randomized Weights: " + maxRandom + "\r\n" + 
            INDENT + "Number of Iterations between saving weights \r\n" + 
            INDENT + "(set to zero or negative to only save after training completes): " + savePeriod;

      printer.print(caseStr + trainingStr);

      System.out.println("Config Updated");
   } //public void writeConfigToFile()

   /**
    * Scans the config file for val. The scanner starts at its current token, reading and skipping over 
    * every subsequent token until it finds a token that matches val. Once that token is scanned, 
    * the scanner's next token is the token which follows the one that matched val.
    *
    * @param val 
    *       The string token to search the config file for. 
    *       If val is not present in the config file after the scanner's current token, 
    *       a NoSuchElementException will be thrown.
    *
    * @return 
    *       The scanner's previously scanned token once it finds val, which is equal to val.
    */
   private String readNextString(String val)
   {
      String nextToken = "";

      while (!nextToken.equals(val))
      {
         nextToken = scanner.next();
      }
      
      return nextToken;
   }

   /**
    * Scans the config file for an integer. The scanner starts at its current token, reading and skipping over 
    * every subsequent token until it finds a token which can be parsed as an integer. The scanner scans that
    * integer token and returns it, and the scanner's next token is the token which follows.
    * If there are no integer tokens present in the config file after the scanner's current token, 
    * a NoSuchElementException will be thrown.
    *
    * @return 
    *       The next integer token in the config file after the scanner's current token
    */
   private int readNextInt()
   {
      while (!scanner.hasNextInt())
         scanner.next();

      return scanner.nextInt(); //Scans next token and parses it as an integer
   }

   /**
    * Scans the config file for a double. The scanner starts at its current token, reading and skipping over 
    * every subsequent token until it finds a token which can be parsed as a double. The scanner scans that
    * double token and returns it, and the scanner's next token is the token which follows.
    * If there are no double tokens present in the config file after the scanner's current token, 
    * a NoSuchElementException will be thrown.
    *
    * @return 
    *       The next double token in the config file after the scanner's current token
    */
   private double readNextDouble()
   {
      while (!scanner.hasNextDouble())
         scanner.next();
      
      return scanner.nextDouble(); //Scans next token and parses it as a double
   }
} //public class ConfigScanner