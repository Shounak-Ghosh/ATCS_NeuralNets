import java.io.File;

/**
 * This class represents a basic Connectivity Model of Perceptrons.
 * 
 * The Model has one input layer, one output layer, and any number of hidden activation layers.
 * Any layer can contain any number of perceptrons. 
 * Any two perceptrons seperated by one connectivity layer can be connected.
 * 
 * All of the model's settings are configurable by creating a config file 
 * (see TemplateConfig.txt in the 'models' folder).
 * 
 * A Model can be created with one of three ways of specifying a config file and a boolean for
 * whether debug prints should occur during execution and training. A String file path to the
 * config file, a File object representing the config file, or a ConfigScanner created with the
 * desired config file all work for the first parameter.
 * 
 * The updateModelFromConfig method calls the ConfigScanner's readConfig method to update all
 * fields from the config file and then sets the Model's fields based on the ConfigScanner's fields.
 * Essentially, the Model's values are reset to the defaults specified in the model config and all
 * intermediate variables are reset. This method is called automatically in the constructor and before 
 * execution and training of the Model.
 * 
 * The thresholdFn returns the threshold function's return value for a given double input. Likewise, the
 * thresholdDerivFn returns the return value of the threshold function's derivative for a given double input.
 * 
 * The Model can be run with inputs using the public run(double[] inputs) method,
 * where the inputs array represents the value for each activation in the input
 * layer to have. The length of the inputs array must be the same as the specified 
 * number.
 * 
 * Viewing the Model's error for a given test case using the Model's current weights can be done
 * with the calculateErrorWithTestCase method, which takes in an integer for which
 * test case to use and returns the error for that test case.
 * 
 * The randomizeWeights method creates random weight values within the range specified in the config file.
 * It uses the createRandomDouble method, which takes in the minimum and maximum double values of the range
 * to generate a random number within and returns the random number.
 * 
 * The Model can be trained using the public train method. The first parameter determines whether the weights
 * should be randomized before training. The second parameter determines whether the weights calculated after 
 * training should be saved to the config file. The third parameter specifies how often the current number of
 * training iterations should be printed. The rest of the configuration options can be modified directly in the
 * model's config file without having to re-compile.
 * 
 * @author Jatin Kohli
 * @since 9/10/19
 */
public class Model 
{
   private static final int INPUT_INDEX = 0; //Index of input activation layer

   private double minLearningFactor;
   private double defaultLearningFactor;
   private double deltaLearningFactor;
   private int savePeriod;

   private boolean isThresholdLinear;

   public ConfigScanner scanner;
   public double[][] activations;
   private double[][][] weights;

   private boolean debug;

   private double[] testCaseInputs;
   private double[] testCaseOutputs;

   private double learningFactor;
   private int numTrainingIterations;
   private int maxIterations;
   private double[][][] deltas; //Stores calculated delta weights for training

   /*
    * Back propogation variables and corresponding symbols in the design.
    *
    * For example, psi[0][1] refers to the psi variable in the second activation layer
    * with index 1. The first activation layer does not have any back propogation variables,
    * so the symbols start with index 0 being the first hidden layer. The last index corresponds
    * to the output layer.
    */
   private double[][] psi;  
   private double[][] theta;
   private double[][] omega;

   private int numInputNodes;
   private int numHiddenLayers;
   private int numHiddenNodes[];
   public int numOutputNodes;

   public int outputIndex;

   private int numWeightLayers;

   /**
    * Creates a Model with the specified number of input perceptrons, 
    * perceptrons for each hidden layer, and output perceptrons from the
    * config file located at configFilePath.
    * 
    * @param configFilePath
    *        The relative or absolute path of the config file for the Model
    * @param debug
    *        True if the model should print all activation values when running
    */
   public Model(String configFilePath, boolean debug)
   {
      this(new ConfigScanner(configFilePath), debug);
   }

   /**
    * Creates a Model with the specified number of input perceptrons, 
    * perceptrons for each hidden layer, and output perceptrons from the
    * File, which must be a valid configFile.
    * 
    * @param configFile
    *        The File representing the config file for the Model
    * @param debug
    *        True if the model should print all activation values when running
    */
   public Model(File configFile, boolean debug)
   {
      this(new ConfigScanner(configFile), debug);
   }

   /**
    * Creates a Model with the specified number of input perceptrons, 
    * perceptrons for each hidden layer, and output perceptrons from the
    * ConfigScanner which has been instantiated with the desired config file.
    * 
    * @param s
    *        The ConfigScanner over the desired Model config file
    * @param debug
    *        True if the model should print all activation values when running
    */
   public Model(ConfigScanner s, boolean debug)
   {
      this.scanner = s;
      this.debug = debug;

      updateModelFromConfig();
   }

   /**
    * Updates the Model based on the configuration files current values. The configFile can be modified, 
    * even while ModelTester is currently running, to update the model without having to re-compile.
    */
   private void updateModelFromConfig()
   {
      scanner.readConfig();

      int maxLayerSize = scanner.maxLayerSize;

      numHiddenLayers = scanner.numHiddenLayers;

      outputIndex = numHiddenLayers + 1;

      activations = new double[numHiddenLayers + 2][maxLayerSize];
      weights = scanner.weights;

      numWeightLayers = weights.length;

      numInputNodes = scanner.inputNodes;
      numHiddenNodes = scanner.hiddenLayerNodes;
      numOutputNodes = scanner.outputNodes;

      isThresholdLinear = scanner.thresholdFunction.equals(scanner.LINEAR_TOKEN); //linear or sigmoid

      minLearningFactor = scanner.minLambda;
      defaultLearningFactor = scanner.defaultLambda;
      deltaLearningFactor = scanner.deltaLambda;
      learningFactor = defaultLearningFactor;
      maxIterations = scanner.maxIterations;
      savePeriod = scanner.savePeriod;

      numTrainingIterations = 0;

      deltas = new double[numWeightLayers][maxLayerSize][maxLayerSize];

      omega = new double[numHiddenLayers + 1][maxLayerSize];
      psi = new double[numHiddenLayers + 1][maxLayerSize];
      theta = new double[numHiddenLayers + 1][maxLayerSize];
   } //private void updateModelFromConfig()

   /**
    * Returns the value of f(input), where f is the threshold function.
    * The threshold function is either the linear function f(x) = x or
    * the sigmoid function f(x) = 1.0 / (1.0 + e^(-x)).
    * 
    * @param input
    *        The input to pass into the threshold function
    * @return
    *        f(input), where f is the threshold function specified in the Model's Config File
    */
   private double thresholdFn(double input)
   {
      double output;

      if (isThresholdLinear) //f(x) = x
      {
         output = input;
      }
      else //f(x) = 1.0 / (1.0 + e^(-x))
      {
         output = 1.0 / (1.0 + Math.exp(-input));
      }

      return output;
   } //private double thresholdFn(double input)

   /**
    * Returns the value of f'(input), where f' is the derivative of the threshold function.
    * The threshold function is either the linear function f(x) = x or
    * the sigmoid function f(x) = 1.0 / (1.0 + e^(-x)).
    * 
    * @param input
    *        The input to pass into the threshold function's derivative
    * @return
    *        f'(input), where f' is the derivative of the threshold function 
    *       specified in the Model's Config File
    */
   private double thresholdDerivFn(double input)
   {
      double output;

      if (isThresholdLinear) //f(x) = x so f'(x) = 1.0
      {
         output = 1.0;
      }
      else //f(x) = 1.0 / (1.0 + e^(-x)) so f'(x) = f(x) * (1.0 - f(x))
      {
         double thresholdFnValue = thresholdFn(input);
         output = thresholdFnValue * (1.0 - thresholdFnValue);
      }

      return output;
   } //private double thresholdDerivFn(double input)

   /**
    * Runs the model with the given set of inputs, updates the values of each perceptron,
    * and returns a double[] with the values of all output perceptrons. This method is
    * only for running and does not update the back propogation variables.
    * 
    * @param inputs 
    *        The values for each input perceptron to have. 
    *        inputs.length must equal the number of input perceptrons in the Model
    *        If inputs.length != the number of activation nodes specified in the config
    *        file, then some inputs may be assumed to be zero
    */
   public void run(double[] inputs)
   {
      activations[INPUT_INDEX] = inputs;

      for (int m = 0; m < numWeightLayers; m++)
      {
         //number of perceptrons in activation layer n = m
         int numJLayerNodes = m == 0 ? numInputNodes : numHiddenNodes[m - 1];

         //number of perceptrons in activation layer n = m + 1
         int numILayerNodes = m == numWeightLayers - 1 ? numOutputNodes : numHiddenNodes[m];

         for (int i = 0; i < numILayerNodes; i++)
         {
            activations[m + 1][i] = 0;

            for (int j = 0; j < numJLayerNodes; j++)
            {
               activations[m + 1][i] += activations[m][j] * weights[m][j][i];
            }

            activations[m + 1][i] = thresholdFn(activations[m + 1][i]);
         }   
      } //for (int m = 0; m < numWeightLayers; m++)
   } //public void run(double[] inputs)

   /**
    * Calculates the error for the given test case in the config file
    * The error for each output is equal to 0.5 * (expectedOutput - actualOutput)^2
    * 
    * @param testCaseNum
    *        The number of the test case to run
    * @return 
    *        A double[] with the errors for each test case in the order they are
    *        listed in the config file
    */
   private double calculateErrorWithTestCase(int testCaseNum)
   {
      testCaseInputs = scanner.testCases[testCaseNum][scanner.TEST_CASE_INPUT_INDEX];
      testCaseOutputs = scanner.testCases[testCaseNum][scanner.TEST_CASE_OUTPUT_INDEX];

      activations[INPUT_INDEX] = testCaseInputs;

      double totalError = 0.0;

      for (int m = 0; m < numWeightLayers; m++)
      {
         //number of perceptrons in activation layer n = m
         int numJLayerNodes = m == 0 ? numInputNodes : numHiddenNodes[m - 1];

         //number of perceptrons in activation layer n = m + 1
         int numILayerNodes = m == numWeightLayers - 1 ? numOutputNodes : numHiddenNodes[m];

         for (int i = 0; i < numILayerNodes; i++)
         {
            activations[m + 1][i] = 0;

            for (int j = 0; j < numJLayerNodes; j++)
            {
               activations[m + 1][i] += activations[m][j] * weights[m][j][i];
            }

            activations[m + 1][i] = thresholdFn(activations[m + 1][i]);

            if (m == numWeightLayers - 1)
            {
               double error = testCaseOutputs[i] - activations[outputIndex][i];
               totalError += error * error;
            }
         } //for (int i = 0; i < numILayerNodes; i++)    
      } //for (int m = 0; m < numWeightLayers; m++)

      return totalError * 0.5; //Multiply by 0.5 to remove unnecessary factor of 2 from derivative
   } //public double calculateErrorWithTestCase(int testCaseNum)

   /**
    * Randomizes weights for training
    *
    * @param min 
    *        The min value to be chosen from
    * @param max 
    *        The max value to be chosen from
    */
   private void randomizeWeights(double min, double max)
   {
      for (int m = 0; m < numWeightLayers; m++)
      {
         //number of perceptrons in activation layer n = m
         int numJLayerNodes = m == 0 ? numInputNodes : numHiddenNodes[m - 1];

         //number of perceptrons in activation layer n = m + 1
         int numILayerNodes = m == numWeightLayers - 1 ? numOutputNodes : numHiddenNodes[m];

         for (int j = 0; j < numJLayerNodes; j++)
            for (int i = 0; i < numILayerNodes; i++)
               weights[m][j][i] = createRandomDouble(min, max);
      } //for (int m = 0; m < numWeightLayers; m++)
   } //private void randomizeWeights(double min, double max)

   /**
    * Generates a random value on the range [min, max) 
    *
    * @param min 
    *        The lower bound of the range, will be included in range
    * @param max 
    *        The upper bound of the range, will not be included in range
    * @return 
    *        A random number on [min, max)
    */
   private double createRandomDouble(double min, double max)
   {
      double range = max - min;
      return (Math.random() * range) + min;
   }

   /**
    * Trains the model using adaptive steepest descent with back propogation.
    * Training ends when the error for all test cases goes below the
    * maximum acceptable error, the number of iterations goes above the 
    * maximum allowable number of iterations, or the learning factor goes
    * to zero. The maximum acceptable error and maximum number of iterations
    * are both specified under the Training section of the config file.
    *
    * @param randomWeights
    *       Whether or not to randomize the weights before training.
    *
    * @param saveWeights
    *       Whether or not to save the weights to the config file after training.
    *
    * @param printPeriod
    *       How often to print the current number of training iterations statements during training. 
    *       If this value is zero or negative, no print statements will occur during training. 
    *       The results of training will be printed after training concludes regardless of the 
    *       printPeriod value.
    *
    * @param printResults
    *       Whether to print the outputs for each test case input with the new set of weights.
    */
   public void train(boolean randomWeights, boolean saveWeights, double printPeriod, boolean printResults)
   {
      boolean saveWeightsDuringTraining = saveWeights && savePeriod > 0;

      if (randomWeights)
         randomizeWeights(scanner.minRandom, scanner.maxRandom);

      double prevError = 0.0;
      double error = 0.0;
      
      boolean shouldPrint = printPeriod > 0; //If user wants print statements

      boolean areAllErrorsAcceptable = false; //False so while loop is entered

      while (!areAllErrorsAcceptable && 
            learningFactor > minLearningFactor && 
            numTrainingIterations < maxIterations)
      {
         areAllErrorsAcceptable = true; //Assume errors for all test cases are below threshold

         for (int testCaseNum = 0; testCaseNum < scanner.testCases.length; testCaseNum++)
         {
            if (shouldPrint && numTrainingIterations % printPeriod == 0)
               System.out.println("Num Iterations: " + numTrainingIterations + "\r\nSign of error change: " + Math.signum(error - prevError));

            testCaseInputs = scanner.testCases[testCaseNum][scanner.TEST_CASE_INPUT_INDEX];
            testCaseOutputs = scanner.testCases[testCaseNum][scanner.TEST_CASE_OUTPUT_INDEX];

            activations[INPUT_INDEX] = testCaseInputs;

            for (int m = 0; m < numWeightLayers; m++) //Forward pass of back propogation
            {
               //number of perceptrons in activation layer n = m
               int numJLayerNodes = m == 0 ? numInputNodes : numHiddenNodes[m - 1];

               //number of perceptrons in activation layer n = m + 1
               int numILayerNodes = m == numWeightLayers - 1 ? numOutputNodes : numHiddenNodes[m];

               for (int i = 0; i < numILayerNodes; i++)
               {
                  omega[m][i] = 0.0;
                  psi[m][i] = 0.0;
                  theta[m][i] = 0.0;

                  for (int j = 0; j < numJLayerNodes; j++)
                  {
                     theta[m][i] += activations[m][j] * weights[m][j][i];
                  }

                  activations[m + 1][i] = thresholdFn(theta[m][i]);

                  if (m == numWeightLayers - 1)
                  {
                     omega[m][i] = testCaseOutputs[i] - activations[outputIndex][i];
                     psi[m][i] = omega[m][i] * thresholdDerivFn(theta[m][i]);
                  }
               } //for (int i = 0; i < numILayerNodes; i++)
            } //for (int m = 0; m < numWeightLayers; m++)

            prevError = 0.0;
            
            for (int i = 0; i < numOutputNodes; i++)
            {
               double outputError = omega[outputIndex - 1][i];
               prevError += outputError * outputError;
            }

            prevError *= 0.5;

            for (int m = numWeightLayers - 1; m >= 0; m--) //Backward pass of back propgation
            {
               //number of perceptrons in activation layer n = m
               int numJLayerNodes = m == 0 ? numInputNodes : numHiddenNodes[m - 1];

               //number of perceptrons in activation layer n = m + 1
               int numILayerNodes = m == numWeightLayers - 1 ? numOutputNodes : numHiddenNodes[m];

               for (int j = 0; j < numJLayerNodes; j++)
               {
                  for (int i = 0; i < numILayerNodes; i++)
                  {
                     if (m > 0) //activation layer m = 0 has no back propogation variables
                        omega[m - 1][j] += psi[m][i] * weights[m][j][i];
                     
                     deltas[m][j][i] = learningFactor * activations[m][j] * psi[m][i];
                     weights[m][j][i] += deltas[m][j][i];
                  }

                  if (m > 0) //activation layer m = 0 has no back propogation variables
                     psi[m - 1][j] = omega[m - 1][j] * thresholdDerivFn(theta[m - 1][j]);
               } //for (int j = 0; j < numJLayerNodes; j++)
            } //for (int m = numWeightLayers - 1; m >= 0; m--)

            error = calculateErrorWithTestCase(testCaseNum);

            if (error > scanner.maxAcceptableError) //If error for test case is too high, training won't end
               areAllErrorsAcceptable = false;
               
            if(deltaLearningFactor != 1) //If delta learning factor is 1, disable adaptive learning
            {
               if (error >= prevError) //if error got worse, rollback weights
               {
                  for (int m = 0; m < numWeightLayers; m++)
                  {
                     //number of perceptrons in activation layer n = m
                     int numJLayerNodes = m == 0 ? numInputNodes : numHiddenNodes[m - 1];

                     //number of perceptrons in activation layer n = m + 1
                     int numILayerNodes = m == numWeightLayers - 1 ? numOutputNodes : numHiddenNodes[m];

                     for (int j = 0; j < numJLayerNodes; j++)
                        for (int i = 0; i < numILayerNodes; i++)
                           weights[m][j][i] -= deltas[m][j][i]; //Rollback weights
                  } //for (int m = 0; m < numWeightLayers; m++)

                  learningFactor /= deltaLearningFactor; //Lower learning factor
               } //if (error >= prevError)
               else
               {
                  learningFactor *= deltaLearningFactor; //Raise learning factor
               }
            } //if(deltaLearningFactor != 1)

            numTrainingIterations++;

            //Save the weights during training is user wants every printPeriod iterations
            if (saveWeightsDuringTraining && numTrainingIterations % savePeriod == 0)
               scanner.writeConfigToFile();

            if (debug)
            {
               System.out.println("DEBUG: Trial num #" + numTrainingIterations);
               System.out.println("\tDEBUG: prevError: " + prevError);
               System.out.println("\tDEBUG: error: " + error);
               System.out.println("\tDEBUG: error decreased: " + (error < prevError));

               System.out.println("\tDEBUG: Deltas:");

               for (int m = 0; m < deltas.length; m++)
               {
                  for (int j = 0; j < deltas[m].length; j++)
                  {
                     for (int i = 0; i < deltas[m][j].length; i++)
                     {
                        if (!(m == numWeightLayers - 1 && i > numOutputNodes - 1))
                        {
                           System.out.println(
                                 "\t\tDEBUG: deltas[" + m + "][" + j + "][" + i + "] = " + deltas[m][j][i]
                           );
                        }
                     }
                  } //for (int j = 0; j < deltas[m].length; j++)
               } //for (int m = 0; m < deltas.length; m++)
            } //if (debug && numTrainingIterations <= 36)
         } // for (int testCaseNum = 0; testCaseNum < scanner.testCases.length; testCaseNum++)
      } // while (!areAllErrorsAcceptable && 
        //       learningFactor > minLearningFactor && 
        //       numTrainingIterations < maxIterations)
      
      if (areAllErrorsAcceptable)
         System.out.println("Training ended, error for all test cases was below the maximum allowable value: " 
               + scanner.maxAcceptableError);
      else if (learningFactor <= minLearningFactor)
         System.out.println("Training ended, learning factor went to zero");
      else if (numTrainingIterations >= scanner.maxIterations)
         System.out.println("Training ended, max number of iterations reached");
      else
         System.out.println("Training ended, reason unknown :(");
      
      System.out.println("Number of Iterations: " + numTrainingIterations);
      System.out.println("Learning Factor: " + learningFactor);

      int testCase = 0;

      if (printResults)
         System.out.println("\r\nInputs/Outputs for each Test Case:");

      while (printResults && testCase < scanner.numTestCases)
      {
         double testCaseError = calculateErrorWithTestCase(testCase);

         double[] inputs = activations[INPUT_INDEX];
         double[] outputs = activations[outputIndex];

         if (printResults)
         {
            System.out.print("\r\nInputs: ");

            for (int j = 0; j < numInputNodes; j++)
               System.out.print(inputs[j] + " ");

            System.out.print("\r\nReturned Outputs: ");

            for (int i = 0; i < numOutputNodes; i++)
               System.out.print(outputs[i] + " ");

            System.out.println("\r\nWith an error of: " + testCaseError);
            
            System.out.println(); //New line between outputs and next test case's inputs
         } //if (printResults)

         testCase++;
      } //while (testCase < scanner.numTestCases)

      if (saveWeights)
         scanner.writeConfigToFile(); //Save Weights to config file after training completes

      //Reset Training Variables for next training
      numTrainingIterations = 0;
      learningFactor = defaultLearningFactor;
   }//public void train(int testCaseNum, double maxAcceptableError)
} //public class Model