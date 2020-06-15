import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.StringTokenizer;

/**
 * The Perceptron is configured via an input file (see specifications below) and computes an output based on
 * initial input activations coupled with the weights between units in a feed-forward fashion. The net output is then
 * compared to the expected output using an error function. After the network is run on all the training sets, the net
 * and mean error is computed. The network can also be trained to converge for a given input file with randomized initial weights.
 * The network is limited to an a input, output and a single hidden layer, and has been optimized by the back-propagation algorithm.
 *
 *
 * INPUT FILE FORMAT(.txt): The input configuration file must be placed within the same directory as the code files
 * An integer N:
 * the total number of layers, including input and output layers
 * A row of N integers separated by whitespaces:
 * The number of units in each layer
 * A row of three integers (A,B,C) separated by whitespaces:
 * (A) The number of training sets, (B) the number of input activations, and (C) the number of output activations
 * An A by (B+C) matrix, with entries separated by whitespaces:
 * Each of the A rows will contain B inputs and C expected outputs, representing the training data
 * An integer W:
 * The number of weights being specified
 * W rows with a singular double value:
 * Each row will have a specific weight value
 * The weights are expected to be entered in order of increasing subscript
 * (ie 000,001,010, 011,100,110 for a 2 2 1 network)
 * If W is smaller than the total number of weights in the network, the remaining weights will be set to 0
 * Following the weights are the hyperparamters of the network:
 * The error threshold
 * The maximum number of iterations allowed
 * The adaptive learning rate
 * The minimum and maximum weight values, in ascending order on the same line
 * An example of proper formatting has been provided below:
 * 3
 * 2 2 1
 * 4 2 1
 * .11 .55 11
 * .22 .66 12
 * .33 .77 13
 * .44 .88 14
 * 6
 * -2.5
 * -2.0
 * -1.5
 * -1.0
 * -0.5
 * 0.25
 * .001
 * 100000
 * 5.0
 * -2.0 2.0
 * @author Shounak Ghosh
 * @version 4.25.2020
 */
public class Perceptron
{
   private int numLayers;              // the total number of layers within the network, including input and output layers
   private int numTrainingSets;        // the number of training sets (activation inputs and expected outputs)
   private int numInputs;              // number of input activations
   private int numOutputs;             // number of output activations
   private int[] layerSizes;           // stores the size of each layer
   private double[][] trainingSet;     // the training data
   private int maxLayerSize = 0;       // the largest layer size
   private double[][] unit;            // stores the activation units
   private double[][] weightedSum;     // the raw weighted sums input into the threshold
   private double[][][] weight;        // stores the weights between units
   private double[][][] delta;         // stores the change in weight values for each weight
   private double LAMBDA;              // the learning factor used to control the magnitude of weight change
   private double MAX_WEIGHT;          // maximum randomized weight value
   private double MIN_WEIGHT;          // minimum randomized weight value
   private int MAX_ITERATIONS;         // maximum number of iterations allowed before training is stopped
   private double ERROR_THRESHOLD;     // the maximum error tolerance allowed after training is complete
   private double[] omega;

   /**
    * Constructor: Creates Perceptron objects
    * @param filename the filename from which the initializing data is read
    */
   public Perceptron(String filename)
   {
      try
      {
         readFile(filename);
      }
      catch (Exception e)
      {
         e.printStackTrace();
      }

      weightedSum = new double[numLayers][maxLayerSize];
      delta = new double[numLayers][maxLayerSize][maxLayerSize];
      omega = new double[maxLayerSize];
   }

   /**
    * Trains the Perceptron to converge under a minimum error threshold or a maximum number of iterations
    */
   public void trainPerceptron()
   {
      printHyperparameters();
      randomizeWeights();

      double error = Double.MAX_VALUE;
      double currentError;
      int numIterations = 0;

      while (numIterations < MAX_ITERATIONS && error > ERROR_THRESHOLD)
      {
         error = 0.0;

         for (int t = 0; t < numTrainingSets; t++)
         {
            for (int input = 0; input < numInputs; input++)
            {
               unit[0][input] = trainingSet[t][input];
            } // load training data

            computeOutput();
            currentError = 0.0;

            for (int I = 0; I < numOutputs; I++)
            {
               currentError += errorFunction(unit[2][I], trainingSet[t][numInputs + I]);
            } // calculate current error

            error = Math.max(error, currentError);
            double psi;

            for (int j = 0; j < layerSizes[1]; j++)
            {
               omega[j] = 0.0;
               for (int i = 0; i < layerSizes[2]; i++)
               {
                  psi = (trainingSet[t][numInputs + i] - unit[2][i]) * thresholdDerivative(weightedSum[2][i]);
                  delta[1][j][i] = LAMBDA * unit[1][j] * psi;
                  omega[j] += psi * weight[1][j][i];
                  weight[1][j][i] += delta[1][j][i];
               }
            } // loop through ji weights

            for (int k = 0; k < layerSizes[0]; k++)
            {
               for (int j = 0; j < layerSizes[1]; j++)
               {
                  delta[0][k][j] = LAMBDA * unit[0][k] * omega[j] * thresholdDerivative(weightedSum[1][j]);
                  weight[0][k][j] += delta[0][k][j];
               }
            } // loop through kj weights

         } // for each training set

         numIterations++;
      } // numIterations < MAX_ITERATIONS && error > ERROR_THRESHOLD

      runTrainingSets();
      System.out.println("Number of iterations: " + numIterations);
   }


   /**
    * Prints out the hyperparameters passed in via the input configuration file
    */
   public void printHyperparameters()
   {
      System.out.println("HYPERPARAMTERS");
      System.out.println("Number of Layers: " + numLayers);
      System.out.println("Number of activations per layer: " + Arrays.toString(layerSizes));
      System.out.println("Error Threshold: " + ERROR_THRESHOLD);
      System.out.println("Maximum Iterations: " + MAX_ITERATIONS);
      System.out.println("Learning factor: " + LAMBDA);
      System.out.println("Min Weight: " + MIN_WEIGHT + " Max Weight: " + MAX_WEIGHT + " Weight Range: " + (MAX_WEIGHT - MIN_WEIGHT));
      System.out.println();
   }

   /**
    * Prints out the delta value for each weight, in ascending order
    */
   public void printDelta()
   {
      System.out.println("DELTAS");
      for (int n = 0; n < numLayers - 1; n++)
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            for (int i = 0; i < layerSizes[n + 1]; i++)
            {
               System.out.println("D: w" + n + "" + "" + j + "" + i + " " + delta[n][j][i]);
            }
         }
      }
      System.out.println();
   }

   /**
    * A utility method for running the network on all of the training sets
    */
   public void runTrainingSets()
   {
      printTrainingSet();
      double error;
      double totalError;

      for (int t = 0; t < numTrainingSets; t++)
      {
         for (int input = 0; input < numInputs; input++)
         {
            unit[0][input] = trainingSet[t][input];
         }

         computeOutput();
         totalError = 0.0;

         for (int I = 0; I < numOutputs; I++)
         {
            error = errorFunction(unit[numLayers - 1][I], trainingSet[t][numInputs + I]);
            totalError += error;
            System.out.println("Output:" + unit[numLayers - 1][I] + " Expected:" + trainingSet[t][numInputs + I] + " Error:" + error);
         } // print each output, expected, and error

         System.out.println("Total Error: " + totalError);
         System.out.println();
      } // for each training set
   }

   /**
    * Calculates the error between the network and expected output
    * @param output   the output computed by the network
    * @param expected the expected training data set output
    * @return the error value; a smaller value corresponds to a lower error
    */
   public double errorFunction(double output, double expected)
   {
      double error = output - expected;
      return 0.5 * error * error;
   }

   /**
    * Sets the specified weight to the given value
    * @param parentLayer the layer from which the weight comes from
    * @param parentUnit  the specific unit from the parent layer
    * @param childUnit   the specific unit from the child layer
    * @param value       the new value of the weight
    */
   public void setWeight(int parentLayer, int parentUnit, int childUnit, double value)
   {
      weight[parentLayer][parentUnit][childUnit] = value;
   }

   /**
    * Randomizes the initial weight values, unless input weights are specified in the input configuration file
    */
   public void randomizeWeights()
   {
      for (int n = 0; n < numLayers - 1; n++)
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            for (int i = 0; i < layerSizes[n + 1]; i++)
            {
               if (weight[n][j][i] == 0.0)
               {
                  weight[n][j][i] = (MAX_WEIGHT - MIN_WEIGHT) * Math.random() + MIN_WEIGHT;
               }
            }
         }
      } // loop through all every weight
   }

   /**
    * Prints out the weights in ascending order
    */
   public void printWeights()
   {
      System.out.println("WEIGHTS");
      for (int n = 0; n < numLayers - 1; n++)
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            for (int i = 0; i < layerSizes[n + 1]; i++)
            {
               System.out.println("w" + n + "" + "" + j + "" + i + " " + weight[n][j][i]);
            }
         }
      } // loop through every weight
      System.out.println();
   }

   public void writeWeights(String filename) throws Exception
   {
      PrintWriter outf = new PrintWriter(new File(filename));
      outf.println("WEIGHTS");
      for (int n = 0; n < numLayers - 1; n++)
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            for (int i = 0; i < layerSizes[n + 1]; i++)
            {
               outf.println("w" + n + "" + "" + j + "" + i + " " + weight[n][j][i]);
            }
         }
      }
      outf.println();
      outf.close();
   }



   /**
    * Computes the output of the network based on the values
    * currently populating the units and weights arrays
    */
   public void computeOutput()
   {
      double weightSum = 0.0;

      for (int k = 1; k < numLayers; k++)
      {
         for (int i = 0; i < layerSizes[k]; i++)
         {
            for (int j = 0; j < layerSizes[k - 1]; j++)
            {
               weightSum += unit[k - 1][j] * weight[k - 1][j][i]; // current unit * weight
            }
            weightedSum[k][i] = weightSum; // store the weighted sum calculated
            unit[k][i] = thresholdFunction(weightSum);
            weightSum = 0.0; // reset the weighted sum
         }
      } // loop through every layer of the network, including activation and output
   } // public void computeOutput()


   /**
    * Prints out the training data set: The inputs and the expected output for each training set
    */
   public void printTrainingSet()
   {
      System.out.println("TRAINING SET");
      for (double[] row : trainingSet)
      {
         System.out.println(Arrays.toString(row));
      }
      System.out.println();
   }

   /**
    * Prints out the activation units currently populating the units array
    */
   public void printUnits()
   {
      System.out.println("ACTIVATION UNITS");
      for (int n = 0; n < numLayers; n++)
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            System.out.println("a" + "" + n + "" + j + " " + unit[n][j]);
         }
      }
   } // public void printUnits()

   /**
    * The function that determines the activation state of the current unit
    * @param x the input into the function, usually the weighted sum of the current unit
    * @return a value representing the activation state of the current unit
    */
   public double thresholdFunction(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } //  public double thresholdFunction(double x)

   /**
    * Evaluates the derivative of the threshold at a given value
    * @param x the value
    * @return the value of the threshold's derivative at x
    */
   public double thresholdDerivative(double x)
   {
      double thresholdFunction = thresholdFunction(x);
      return thresholdFunction * (1.0 - thresholdFunction);
   } // public double thresholdDerivative(double x)

   /**
    * Reads in the preliminary input file containing the
    * hyper-parameters necessary to configure the network
    * @param filename the name of the input file being read from
    * @throws Exception in the case of an input/output error
    */
   public void readFile(String filename) throws Exception
   {
      BufferedReader f = new BufferedReader(new FileReader(filename));
      StringTokenizer st = new StringTokenizer(f.readLine());
      numLayers = Integer.parseInt(st.nextToken());
      layerSizes = new int[numLayers];

      st = new StringTokenizer(f.readLine());
      for (int n = 0; n < numLayers; n++)
      {
         layerSizes[n] = Integer.parseInt(st.nextToken());
         if (layerSizes[n] > maxLayerSize)
         {
            maxLayerSize = layerSizes[n];
         }
      }

      st = new StringTokenizer(f.readLine());
      numTrainingSets = Integer.parseInt(st.nextToken());
      numInputs = Integer.parseInt(st.nextToken());
      numOutputs = Integer.parseInt(st.nextToken());
      trainingSet = new double[numTrainingSets][numInputs + numOutputs];
      double val;

      for (int n = 0; n < numTrainingSets; n++)
      {
         st = new StringTokenizer(f.readLine());
         for (int j = 0; j < numInputs + numOutputs; j++)
         {
            val = Double.parseDouble(st.nextToken());
            trainingSet[n][j] = val;
         }
      } // loop through each training set

      unit = new double[numLayers][maxLayerSize];
      weight = new double[numLayers][maxLayerSize][maxLayerSize];

      st = new StringTokenizer(f.readLine());
      int numWeights = Integer.parseInt(st.nextToken());

      for (int n = 0; n < numLayers - 1; n++)
      {
         for (int j = 0; j < layerSizes[n]; j++)
         {
            for (int i = 0; i < layerSizes[n + 1]; i++)
            {
               if (numWeights > 0)
               {
                  st = new StringTokenizer(f.readLine());
                  weight[n][j][i] = Double.parseDouble(st.nextToken());
                  numWeights--;
               }
            }
         }
      } // loop through every weight

      st = new StringTokenizer(f.readLine());
      ERROR_THRESHOLD = Double.parseDouble(st.nextToken());

      st = new StringTokenizer(f.readLine());
      MAX_ITERATIONS = Integer.parseInt(st.nextToken());

      st = new StringTokenizer(f.readLine());
      LAMBDA = Double.parseDouble(st.nextToken());

      st = new StringTokenizer(f.readLine());
      MIN_WEIGHT = Double.parseDouble(st.nextToken());
      MAX_WEIGHT = Double.parseDouble(st.nextToken());
      f.close();
   } // public void readFile(String filename) throws Exception
} // public class Perceptron

