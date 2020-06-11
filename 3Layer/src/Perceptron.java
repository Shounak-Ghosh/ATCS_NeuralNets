import java.io.*;
import java.util.*;

/**
 * The Perceptron is an n-layer neural network that implements the back-propagation algorithm.
 * It solves boolean algebra and is configured using the "master.txt" input file along with a set of training files,
 * which outline the expected set of outputs for a given set of inputs.
 *
 * TABLE OF CONTENTS
 * Perceptron(String filename) throws IOException
 * static void main(String[] args) throws IOException
 * double computeError(int trainingSet)
 * void computeOutput()
 * double getMaxError()
 * void loadTrainingSets(String inFile) throws IOException
 * void loadWeights(String weightFile) throws IOException
 * void printWeights()
 * void randomizeWeights()
 * void runTrainingSets()
 * double thresholdDerivative(double thresholdFunction)
 * void trainPerceptron()
 *
 * @author Shounak Ghosh
 * @version 5.02.2020
 *
 */
public class Perceptron
{
   private double MIN_WEIGHT;     // The lowest possible random weight.
   private double MAX_WEIGHT;     // The highest possible random weight.
   private double LAMBDA;         // The value by which the delta for each weight is amplified.
   private double errorThreshold; // Maximum error allowed for a training set.
   private int MAX_ITERATIONS;    // The maximum number of iterations allowed before ending training.
   private double[][][] weights;  // Stores the weights between units
   private double[][] units;      // Stores the values of the units
   private double[][] weightedSums;
   private double[][] input;      // The network inputs taken from the input file
   private double[][] output;     // The expected inputs taken from the input file
   private double[][] psi;        // Psi values updated over the course of training
   private int[] layerSizes;      // Stores the size of each layer, including input and output
   private int numTrainingSets;   // The number of training sets present in the input file
   private int totalLayers;       // The total number of layers including input, output, and hidden layers
   private int iterations;        // The number of iterations taken by the perceptron to converge
   private double error;          // the maximum error present after the perceptron converged
   /**
    * This is the constructor for the neural network
    * @param filename the name of the input file from which network inputs and expected outputs can be found
    * @throws IOException if the file called for is not found
    */
   public Perceptron(String filename) throws IOException
   {
      String configurationFile = "master.txt";
      String inputFileName;
      int maxUnits;
      BufferedReader f = new BufferedReader(new FileReader(new File("files/" + configurationFile)));
      int numHiddenLayers = Integer.parseInt(f.readLine());
      totalLayers = numHiddenLayers + 2;
      StringTokenizer st = new StringTokenizer(f.readLine());
      int numInputs = Integer.parseInt(st.nextToken());
      maxUnits = numInputs;
      layerSizes = new int[totalLayers];
      layerSizes[0] = numInputs;

      for (int layer = 1; layer < totalLayers - 1; layer++)
      {
         layerSizes[layer] = Integer.parseInt(st.nextToken());

         maxUnits = Math.max(maxUnits, layerSizes[layer]);
      }

      layerSizes[totalLayers - 1] = Integer.parseInt(st.nextToken());

      maxUnits = Math.max(maxUnits, layerSizes[totalLayers - 1]);

      MIN_WEIGHT = Double.parseDouble(f.readLine());
      MAX_WEIGHT = Double.parseDouble(f.readLine());
      LAMBDA = Double.parseDouble(f.readLine());
      errorThreshold = Double.parseDouble(f.readLine());
      MAX_ITERATIONS = Integer.parseInt(f.readLine());
      inputFileName = f.readLine();
      f.close();

      weights = new double[totalLayers - 1][maxUnits][maxUnits];
      units = new double[totalLayers][maxUnits];
      weightedSums = new double[totalLayers][maxUnits];
      psi = new double[totalLayers][maxUnits];
      randomizeWeights();

      if (filename.length() > 0)
      {
         inputFileName = filename;
      }
      loadTrainingSets(inputFileName);


      System.out.println("HYPERPARAMETERS");
      System.out.println("Number of layers: " + layerSizes.length);
      System.out.println("Number of activations per layer: " + Arrays.toString(layerSizes));
      System.out.println("Error Threshold: " + errorThreshold);
      System.out.println("Max Iterations: " + MAX_ITERATIONS);
      System.out.println("Learning Factor: " + LAMBDA);
      System.out.println("Min Weight: " + MIN_WEIGHT + " Max Weight: " + MAX_WEIGHT + " Weight Range: " + Math.abs(MAX_WEIGHT - MIN_WEIGHT));
      System.out.println();
      System.out.println("TRAINING SETS");

      for (int i = 0; i < numTrainingSets; i++)
      {
         System.out.println(Arrays.toString(input[i]) + Arrays.toString(output[i]));
      }

      trainPerceptron();
      runTrainingSets();

      System.out.println();
      System.out.println("Number of Iterations: " + iterations);
      System.out.println("Maximum Error: " + error);

   }

   /**
    * Driver method; used for testing the Perceptron
    * @param args the name of the file to be used for input
    *             if left empty, the default filename in "master.txt" will be used
    * @throws IOException thrown if there is a file input error
    */
   public static void main(String[] args) throws IOException
   {
      if (args.length == 1)
      {
         new Perceptron(args[0]);
      }
      else
      {
         new Perceptron("");
      }

   }

   /**
    * Computes the error for a given training set
    * @param trainingSet the trainingSet for the total error .5 * (sum of error per output ^ 2)
    * @return the total error for the given trainingSet
    */
   public double computeError(int trainingSet)
   {
      double maxError = 0.0;

      for (int outputs = 0; outputs < layerSizes[totalLayers - 1]; outputs++)
         maxError += (output[trainingSet][outputs] - units[totalLayers - 1][outputs])
                 * (output[trainingSet][outputs] - units[totalLayers - 1][outputs]);

      return .5 * maxError;
   } //  public double computeError(int trainingSet)

   /**
    * Evaluates each unit in the perceptron using the current network inputs
    */
   public void computeOutput()
   {
      double sum = 0.0;

      for (int layer = 1; layer < totalLayers; layer++)
      {
         for (int childUnit = 0; childUnit < layerSizes[layer]; childUnit++)
         {
            sum = 0.0;

            for (int parentUnit = 0; parentUnit < layerSizes[layer - 1]; parentUnit++)
            {
               sum += units[layer - 1][parentUnit] * weights[layer - 1][parentUnit][childUnit];
            }
            weightedSums[layer][childUnit] = units[layer][childUnit];
            units[layer][childUnit] = thresholdFunction(sum);
         }
      } //  for (int layer = 1; layer < totalLayers; layer++)
   } // public void computeOutput()

   /**
    * Fetches the largest error over all training sets
    * @return the maximum error
    */
   public double getMaxError()
   {
      units[0] = input[0];
      computeOutput();
      double maxError = computeError(0);
      double curError;

      for (int trial = 1; trial < numTrainingSets; trial++)
      {
         units[0] = input[trial];
         computeOutput();
         curError = computeError(trial);

         if (curError > maxError)
         {
            maxError = curError;
         }
      }

      return maxError;
   } // public double getMaxError()

   /**
    * Takes in the training set data: network inputs and expected outputs from a .txt file
    * @param inFile the filename
    * @throws IOException thrown if the file is not found
    */
   public void loadTrainingSets(String inFile) throws IOException
   {
      BufferedReader f = new BufferedReader(new FileReader(new File("files/" + inFile)));
      numTrainingSets = Integer.parseInt(f.readLine());

      input = new double[numTrainingSets][layerSizes[0]];
      output = new double[numTrainingSets][layerSizes[totalLayers - 1]];
      StringTokenizer st;

      for (int set = 0; set < numTrainingSets; set++)
      {
         st = new StringTokenizer(f.readLine());

         for (int inputs = 0; inputs < layerSizes[0]; inputs++)
         {
            input[set][inputs] = Double.parseDouble(st.nextToken());
         }

         for (int outputs = 0; outputs < layerSizes[totalLayers - 1]; outputs++)
         {
            output[set][outputs] = Double.parseDouble(st.nextToken());
         }
      } // for (int set = 0; set < numTrainingSets; set++)

      f.close();
   } // public void loadTrainingSets(String inFile) throws IOException

   /**
    * Takes in custom weights from a .txt file.
    * @param weightFile the filename
    * @throws IOException thrown if the file is not found
    */
   public void loadWeights(String weightFile) throws IOException
   {
      BufferedReader weightsReader = new BufferedReader(new FileReader(new File("files/" + weightFile)));

      for (int layer = 0; layer < totalLayers - 1; layer++)
      {
         for (int parentUnit = 0; parentUnit < layerSizes[layer]; parentUnit++)
         {
            for (int childUnit = 0; childUnit < layerSizes[layer + 1]; childUnit++)
            {
               weights[layer][parentUnit][childUnit] = Double.parseDouble(weightsReader.readLine());
            }
         }
      }
      weightsReader.close();
   } // public void loadWeights(String weightFile) throws IOException

   /**
    * Prints the weights between units in the perceptron
    */
   private void printWeights()
   {
      for (int layer = 0; layer < totalLayers - 1; layer++) // Prints out weights
      {
         for (int childUnit = 0; childUnit < layerSizes[layer + 1]; childUnit++)
         {
            for (int parentUnit = 0; parentUnit < layerSizes[layer]; parentUnit++)
            {
               System.out.println("Weight " + layer + " " + parentUnit + " " + childUnit + " : " + weights[layer][parentUnit][childUnit]);
            }
         }
      }
   } // private void printWeights()

   /**
    * This randomizes the weights of the perceptron between two values, with bounds specified by MIN_WEIGHT
    * and MAX_WEIGHT.
    */
   public void randomizeWeights()
   {
      for (int layer = 1; layer < totalLayers; layer++)
      {
         for (int parentUnit = 0; parentUnit < layerSizes[layer - 1]; parentUnit++)
         {
            for (int childUnit = 0; childUnit < layerSizes[layer]; childUnit++)
            {
               weights[layer - 1][parentUnit][childUnit] = (Math.random() * (MAX_WEIGHT - MIN_WEIGHT)) + MIN_WEIGHT;
            }
         }
      }
   } //  public void randomizeWeights()

   /**
    * This prints the test cases of the network, it runs the input into the network and prints out the output.
    */
   private void runTrainingSets()
   {
      for (int i = 0; i < numTrainingSets; i++)
      {
         System.out.println("");
         units[0] = input[i];
         computeOutput();

         for (int j = 0; j < layerSizes[totalLayers - 1]; j++)
         {
            System.out.println("Output: " + units[totalLayers - 1][j] + " Expected: " + output[i][j]);
         }
         System.out.println("Error: " + computeError(i));
      } // for (int i = 0; i < numTrainingSets; i++)
   } // private void runTrainingSets()

   /**
    * Evaluates the derivative of the threshold for a given value
    * @param thresholdFunction the value, value = threshold(x) for some x
    * @return the value of the threshold's derivative at thresholdFunction
    */
   public double thresholdDerivative(double thresholdFunction)
   {
      return thresholdFunction * (1.0 - thresholdFunction);
   } // public double thresholdDerivative(double x)

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
    * This trains the network to the test cases. This requires that all weights and inputs created already.
    */
   public void trainPerceptron()
   {
      int numIterations = 0;
      double maxError = getMaxError();
      double curError;

      while (maxError > errorThreshold && numIterations < MAX_ITERATIONS)
      {
         for (int trainingSet = 0; trainingSet < numTrainingSets; trainingSet++)
         {
            units[0] = input[trainingSet];
            computeOutput();

            for (int index = 0; index < layerSizes[totalLayers - 1]; index++)
            {
               psi[totalLayers - 1][index] = (output[trainingSet][index] - units[totalLayers - 1][index])
                       * thresholdDerivative(units[totalLayers - 1][index]);
            }

            double omega;
            for (int layer = totalLayers - 2; layer >= 0; layer--)
            {
               for (int parentUnit = 0; parentUnit < layerSizes[layer]; parentUnit++)
               {
                  omega = 0.0;
                  for (int childUnit = 0; childUnit < layerSizes[layer + 1]; childUnit++)
                  {
                     omega += psi[layer + 1][childUnit] * weights[layer][parentUnit][childUnit];
                     weights[layer][parentUnit][childUnit] += LAMBDA * units[layer][parentUnit]
                             * psi[layer + 1][childUnit];
                  }
                  psi[layer][parentUnit] = omega * thresholdDerivative(units[layer][parentUnit]);
               } // for (int parentUnit = 0; parentUnit < layerSizes[layer]; parentUnit++)
            } //  for (int layer = totalLayers - 2; layer >= 0; layer--)
         } // for (int trainingSet = 0; trainingSet < numTrainingSets; trainingSet++)

         curError = getMaxError();
         maxError = curError;
         numIterations++;
      } // while (maxError > ERROR_THRESHOLD && numIterations++ < MAXIMUM_NUMBER_OF_ITERATION)

      iterations = numIterations;
      error = maxError;
   }
} // public class Perceptron